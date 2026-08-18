# SPDX-License-Identifier: Apache-2.0
"""
KV cache for autoregressive decoding.

Without one, generating N tokens re-runs attention over the whole prefix at
every step: the keys and values for tokens already seen are recomputed from
scratch N times, so the cost grows with the square of the sequence rather than
linearly. Caching them turns each step into one new column of K and V plus an
attention pass whose Q is a single row.

The cache is allocated once at `max_seq_len` and filled in place, so decoding
does no allocation and no concatenation. Under GQA it holds `n_kv_head` heads,
not `n_head` - the same sharing that makes GQA cheap in the kernel makes the
cache 4x smaller for Qwen3 and 16x smaller for Muse Glimmer.
"""

import cupy as cp


class KVCache:
    """Preallocated per-layer key/value cache.

    Example:
        cache = KVCache(n_layer=32, batch=1, n_kv_head=2, max_seq_len=4096,
                        head_dim=128, dtype=cp.dtype('bfloat16'))

        k, v = cache.append(layer_idx, k_new, v_new)   # returns history so far
        attn = cutile_causal_attention(q, k, v, n_head, n_kv_head)
        ...
        cache.reset()
    """

    def __init__(
        self,
        n_layer: int,
        batch: int,
        n_kv_head: int,
        max_seq_len: int,
        head_dim: int,
        dtype=cp.float32,
    ):
        if n_layer <= 0 or batch <= 0 or n_kv_head <= 0 or max_seq_len <= 0:
            raise ValueError("cache dimensions must be positive")

        self.n_layer = n_layer
        self.batch = batch
        self.n_kv_head = n_kv_head
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.dtype = dtype

        shape = (n_layer, batch, n_kv_head, max_seq_len, head_dim)
        self.k = cp.zeros(shape, dtype=dtype)
        self.v = cp.zeros(shape, dtype=dtype)
        self.length = 0

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes

    def reset(self) -> None:
        """Forget the history. The buffers stay allocated."""
        self.length = 0

    def append(self, layer_idx: int, k_new: cp.ndarray, v_new: cp.ndarray):
        """Write one step's K/V for a layer and return the history including it.

        The returned arrays are views into the cache, valid until the next
        append that advances past them.

        Args:
            layer_idx: Which layer's slice to write
            k_new: (batch, n_kv_head, new_len, head_dim)
            v_new: same shape as k_new

        Returns:
            (k, v), each (batch, n_kv_head, length + new_len, head_dim)
        """
        if not 0 <= layer_idx < self.n_layer:
            raise IndexError(
                f"layer_idx {layer_idx} out of range for {self.n_layer} layers"
            )
        if k_new.shape != v_new.shape:
            raise ValueError(f"K and V shapes differ: {k_new.shape} vs {v_new.shape}")

        batch, n_kv_head, new_len, head_dim = k_new.shape
        if (batch, n_kv_head, head_dim) != (self.batch, self.n_kv_head, self.head_dim):
            raise ValueError(
                f"expected (batch={self.batch}, n_kv_head={self.n_kv_head}, "
                f"*, head_dim={self.head_dim}), got {k_new.shape}"
            )

        end = self.length + new_len
        if end > self.max_seq_len:
            raise ValueError(
                f"cache overflow: {self.length} + {new_len} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        self.k[layer_idx, :, :, self.length:end] = k_new
        self.v[layer_idx, :, :, self.length:end] = v_new

        # Advancing on the last layer keeps every layer writing at the same
        # offset within a step, so callers do not have to sequence the update.
        if layer_idx == self.n_layer - 1:
            self.length = end

        # These are strided views into the fixed-capacity cache.  cuTile arrays
        # carry strides, so attention can read them directly; making them
        # contiguous here copied the entire history twice at every decode step.
        return (
            self.k[layer_idx, :, :, :end],
            self.v[layer_idx, :, :, :end],
        )

    def reserve(self, layer_idx: int, new_len: int):
        """Expose write slots and history views for a fused producer kernel."""
        if not 0 <= layer_idx < self.n_layer:
            raise IndexError(
                f"layer_idx {layer_idx} out of range for {self.n_layer} layers"
            )
        if new_len <= 0:
            raise ValueError(f"new_len must be positive, got {new_len}")
        end = self.length + new_len
        if end > self.max_seq_len:
            raise ValueError(
                f"cache overflow: {self.length} + {new_len} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )
        start = self.length
        if layer_idx == self.n_layer - 1:
            self.length = end
        return (
            self.k[layer_idx, :, :, start:end],
            self.v[layer_idx, :, :, start:end],
            self.k[layer_idx, :, :, :end],
            self.v[layer_idx, :, :, :end],
        )

    def __repr__(self) -> str:
        return (
            f"KVCache(n_layer={self.n_layer}, batch={self.batch}, "
            f"n_kv_head={self.n_kv_head}, length={self.length}/{self.max_seq_len}, "
            f"head_dim={self.head_dim}, dtype={self.dtype}, "
            f"{self.nbytes / 2**20:.1f} MiB)"
        )
