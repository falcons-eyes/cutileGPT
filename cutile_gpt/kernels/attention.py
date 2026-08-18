# SPDX-License-Identifier: Apache-2.0
"""
Causal Self-Attention for cutile GPT.

Optimized with:
- num_ctas and occupancy hints
- latency hints for memory access
- flush_to_zero and approx rounding for perf
- Larger tile sizes for better occupancy
"""

import math

import cuda.tile as ct
import cupy as cp
import numpy as np
from cuda.tile import RoundingMode as RMd

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

INV_LOG_2 = 1.0 / math.log(2)


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def causal_attention_kernel(
    Q, K, V, Out,
    qk_scale: float,
    q_offset: int,
    TILE_D: ConstInt,
    N_HEAD: ConstInt,
    N_KV_HEAD: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    WINDOW: ConstInt
):
    """
    Optimized causal multi-head self-attention kernel.

    Args:
        Q: Query tensor (batch, n_head, seq_len, head_dim)
        K: Key tensor (batch, n_kv_head, seq_len, head_dim)
        V: Value tensor (batch, n_kv_head, seq_len, head_dim)
        Out: Output tensor (batch, n_head, seq_len, head_dim)
        qk_scale: Scale factor (1/sqrt(head_dim))
        TILE_D: Head dimension, rounded up to a power of two. Lanes past the
            real head_dim load as zero and are clipped on store.
        q_offset: Absolute position of the first query row. Zero when Q and K
            span the same range; equal to the cached length when decoding, where
            Q holds only the new tokens but K/V hold the whole history. Runtime
            rather than compile-time, since it changes every decode step.
        N_HEAD: Number of query heads
        N_KV_HEAD: Number of key/value heads. Equal to N_HEAD for plain MHA;
            smaller for GQA, where each KV head is shared by N_HEAD/N_KV_HEAD
            query heads. Qwen3 uses 32/8, Muse Glimmer 32/2.
        TILE_M: Tile size for query sequence
        TILE_N: Tile size for key/value sequence
        WINDOW: Sliding window width, or 0 for full causal attention. A query
            at position p then attends to (p - WINDOW, p]. Gemma alternates
            local and global layers 5:1, Muse Glimmer 3:1 at width 2048.
    """
    bid_x = ct.bid(0)  # Query tile index
    bid_y = ct.bid(1)  # Batch * head index

    batch_idx = bid_y // N_HEAD
    head_idx = bid_y % N_HEAD

    # Grouped-query attention lives entirely in this index. Every query head
    # keeps its own tile of Q, but neighbouring heads read the same K and V,
    # so the KV cache shrinks by N_HEAD/N_KV_HEAD. With N_KV_HEAD == N_HEAD
    # it reduces to head_idx and the kernel is plain MHA again.
    kv_head_idx = head_idx // (N_HEAD // N_KV_HEAD)

    # Scale for exp2 optimization
    qk_scale_log2 = qk_scale * INV_LOG_2

    # Query positions are absolute, so a cached decode step masks against the
    # history rather than against its own tiny Q.
    offs_m = q_offset + bid_x * TILE_M + ct.arange(TILE_M, dtype=np.int32)
    offs_m = offs_m[:, None]

    # Key/Value position offsets
    offs_n_tile = ct.arange(TILE_N, dtype=np.int32)
    offs_n_tile = offs_n_tile[None, :]

    # Online softmax accumulators
    m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)

    # Load query tile with latency hint and TMA
    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0),
                shape=(1, 1, TILE_M, TILE_D),
                padding_mode=ct.PaddingMode.ZERO,
                latency=4, allow_tma=True).reshape((TILE_M, TILE_D))

    # Causal masking: only attend to positions <= current position
    seq_len = K.shape[2]
    m_end = q_offset + (bid_x + 1) * TILE_M
    Tc = ct.cdiv(min(m_end, seq_len), TILE_N)

    # A windowed layer skips the KV tiles that fall entirely outside the
    # window instead of loading and masking them, which is where the saving
    # is. WINDOW is a compile-time constant, so a global layer compiles this
    # branch away and starts at 0 as before.
    if WINDOW > 0:
        m_start = q_offset + bid_x * TILE_M
        j_start = max(0, (m_start - WINDOW + 1) // TILE_N)
    else:
        j_start = 0

    # Loop over K, V blocks
    for j in range(j_start, Tc):
        # Load K tile (transposed for matmul) with latency hint and TMA
        k = ct.load(K, index=(batch_idx, kv_head_idx, 0, j),
                    shape=(1, 1, TILE_D, TILE_N),
                    order=(0, 1, 3, 2),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=2, allow_tma=True).reshape((TILE_D, TILE_N))

        # QK^T
        qk = ct.full((TILE_M, TILE_N), 0., dtype=np.float32)
        qk = ct.mma(q, k, qk)

        # Apply causal mask, plus the window's lower bound when there is one
        offs_n = j * TILE_N + offs_n_tile
        mask = offs_m >= offs_n
        if WINDOW > 0:
            mask = mask & (offs_n > offs_m - WINDOW)
        mask = ct.where(mask, 0.0, -np.inf)
        qk += mask

        # Online softmax.
        #
        # A windowed layer can hand a row a tile that is masked end to end -
        # row 191 with WINDOW=64 sees nothing in the tile covering keys 64..127
        # - and then the running max is still -inf. Subtracting it would give
        # -inf - -inf = NaN, so the max is pinned to 0 for exactly that case:
        # every score in the tile is -inf, exp2 takes them to 0, and alpha
        # takes the (empty) accumulator to 0 as well, which is what an empty
        # tile should contribute. Plain causal attention never reaches this,
        # since a row always sees at least its own diagonal.
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2)
        if WINDOW > 0:
            m_ij = ct.where(m_ij == -np.inf, 0.0, m_ij)
        qk = qk * qk_scale_log2 - m_ij
        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # Load V and accumulate with latency hint and TMA
        v = ct.load(V, index=(batch_idx, kv_head_idx, j, 0),
                    shape=(1, 1, TILE_N, TILE_D),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=4, allow_tma=True).reshape((TILE_N, TILE_D))
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    # Final normalization with approximate division for performance
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


def cutile_causal_attention(
    q: cp.ndarray,
    k: cp.ndarray,
    v: cp.ndarray,
    n_head: int,
    n_kv_head: int | None = None,
    q_offset: int | None = None,
    window: int = 0,
) -> cp.ndarray:
    """
    Compute causal self-attention, multi-head or grouped-query.

    This function expects Q, K, V already projected and reshaped to
    (batch, n_head, q_len, head_dim) and (batch, n_kv_head, kv_len, head_dim).
    q_len and kv_len differ when decoding against a cache.

    Args:
        q: Query tensor (batch, n_head, q_len, head_dim)
        k: Key tensor (batch, n_kv_head, kv_len, head_dim)
        v: Value tensor (batch, n_kv_head, kv_len, head_dim)
        n_head: Number of query heads
        n_kv_head: Number of key/value heads. Defaults to K's head count, which
            is plain MHA when it equals n_head. Pass a smaller divisor of
            n_head for GQA - Qwen3 uses 32/8, Muse Glimmer 32/2, shrinking the
            KV cache 4x and 16x.
        q_offset: Absolute position of the first query row. Defaults to
            kv_len - q_len, which is right for both a full prefill (0) and a
            decode step against a cache.
        window: Sliding window width, or 0 for full causal attention. Tiles
            that fall entirely outside the window are skipped rather than
            masked. Gemma alternates local and global layers 5:1, Muse Glimmer
            3:1 at width 2048.

    Returns:
        Attention output (batch, n_head, q_len, head_dim)
    """
    if not all(isinstance(t, cp.ndarray) for t in (q, k, v)):
        raise ValueError("Tensors must be CuPy arrays on CUDA device")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K, and V must all have shape (batch, head, seq, dim)")

    batch, q_heads, seq_len, head_dim = q.shape
    kv_len = k.shape[2]
    if seq_len <= 0 or kv_len <= 0:
        raise ValueError("Q and K/V sequence lengths must be positive")
    if q_heads != n_head:
        raise ValueError(f"Q carries {q_heads} heads but n_head={n_head}")
    if k.shape[0] != batch or v.shape[0] != batch:
        raise ValueError(
            f"Q/K/V batch sizes differ: {batch}, {k.shape[0]}, {v.shape[0]}"
        )
    if k.shape[3] != head_dim or v.shape[3] != head_dim:
        raise ValueError(
            f"Q/K/V head dimensions differ: {head_dim}, {k.shape[3]}, {v.shape[3]}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"Q/K/V dtypes differ: {q.dtype}, {k.dtype}, {v.dtype}")

    if q_offset is None:
        q_offset = kv_len - seq_len
    if q_offset < 0:
        raise ValueError(
            f"q_offset must be >= 0, got {q_offset} "
            f"(q_len={seq_len} exceeds kv_len={kv_len})"
        )
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")

    if n_kv_head is None:
        n_kv_head = k.shape[1]
    if n_head % n_kv_head != 0:
        raise ValueError(
            f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})"
        )
    if k.shape[1] != n_kv_head or v.shape[1] != n_kv_head:
        raise ValueError(
            f"K and V must have n_kv_head={n_kv_head} heads, "
            f"got K={k.shape[1]} V={v.shape[1]}"
        )
    if v.shape[2] != kv_len:
        raise ValueError(f"K and V lengths differ: {kv_len} vs {v.shape[2]}")

    # Scale factor
    qk_scale = 1.0 / math.sqrt(head_dim)

    # Always return BHSD-contiguous storage.  Inputs may be strided views
    # (notably QKV transposes and the fixed-capacity KV cache); cuTile consumes
    # their strides directly instead of materializing layout copies.
    out = cp.empty(q.shape, dtype=q.dtype)

    # Decode usually has one query row.  Compiling it as a 64-row tile wastes
    # almost all MMA/reduction work, so specialize short query lengths while
    # retaining 64 rows for throughput-oriented prefill.
    tile_m = min(64, 1 << (seq_len - 1).bit_length())
    tile_n = 64
    # cutile needs power-of-two tile dims. head_dim usually is one, but Phi
    # uses 96 - rounding up leaves the extra lanes zero-padded on load and
    # clipped on store, and they contribute nothing to the QK product.
    tile_d = 1 << (head_dim - 1).bit_length()

    # Grid dimensions
    grid_x = math.ceil(seq_len / tile_m)
    grid_y = batch * n_head

    ct.launch(
        cp.cuda.get_current_stream(),
        (grid_x, grid_y, 1),
        causal_attention_kernel,
        (q, k, v, out, qk_scale, q_offset, tile_d, n_head, n_kv_head,
         tile_m, tile_n, window)
    )

    return out


def cutile_mha_forward(
    x: cp.ndarray,
    c_attn_weight: cp.ndarray,
    c_attn_bias: cp.ndarray,
    c_proj_weight: cp.ndarray,
    c_proj_bias: cp.ndarray,
    n_head: int,
    c_attn_weight_t: cp.ndarray = None,
    c_proj_weight_t: cp.ndarray = None
) -> cp.ndarray:
    """
    Full multi-head attention forward pass (matching minGPT).

    Args:
        x: Input tensor (batch, seq_len, n_embd)
        c_attn_weight: Combined QKV projection weight (3*n_embd, n_embd)
        c_attn_bias: Combined QKV projection bias (3*n_embd,)
        c_proj_weight: Output projection weight (n_embd, n_embd)
        c_proj_bias: Output projection bias (n_embd,)
        n_head: Number of attention heads
        c_attn_weight_t: Optional pre-transposed c_attn_weight
        c_proj_weight_t: Optional pre-transposed c_proj_weight

    Returns:
        Output tensor (batch, seq_len, n_embd)
    """
    from .linear import cutile_linear_bias

    batch, seq_len, n_embd = x.shape
    head_dim = n_embd // n_head

    # Combined QKV projection
    qkv = cutile_linear_bias(x, c_attn_weight, c_attn_bias, c_attn_weight_t)  # (B, T, 3*n_embd)

    # Split into Q, K, V
    q, k, v = cp.split(qkv, 3, axis=2)

    # Reshape to (batch, n_head, seq_len, head_dim)
    q = cp.transpose(cp.reshape(q, (batch, seq_len, n_head, head_dim)), (0, 2, 1, 3))
    k = cp.transpose(cp.reshape(k, (batch, seq_len, n_head, head_dim)), (0, 2, 1, 3))
    v = cp.transpose(cp.reshape(v, (batch, seq_len, n_head, head_dim)), (0, 2, 1, 3))

    # Attention
    y = cutile_causal_attention(q, k, v, n_head)

    # Reshape back: (batch, n_head, seq_len, head_dim) -> (batch, seq_len, n_embd)
    y = cp.transpose(y, (0, 2, 1, 3))
    if not y.flags.c_contiguous:
        y = cp.ascontiguousarray(y)
    y = cp.reshape(y, (batch, seq_len, n_embd))

    # Output projection
    y = cutile_linear_bias(y, c_proj_weight, c_proj_bias, c_proj_weight_t)

    return y


# Reference CuPy implementation
def cupy_causal_attention(q, k, v, n_head):
    """CuPy reference causal attention"""
    batch, n_head, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    # QK^T
    att = cp.matmul(q, cp.transpose(k, (0, 1, 3, 2))) * scale

    # Causal mask
    mask = cp.tril(cp.ones((seq_len, seq_len)))
    att = cp.where(mask == 0, float('-inf'), att)

    # Softmax
    att = cp.exp(att - cp.max(att, axis=-1, keepdims=True))
    att = att / cp.sum(att, axis=-1, keepdims=True)

    # Weighted sum
    y = cp.matmul(att, v)
    return y


if __name__ == "__main__":
    print("--- Testing cutile Causal Attention kernel ---")

    batch, n_head, seq_len, head_dim = 2, 3, 32, 16
    n_embd = n_head * head_dim

    q = cp.random.randn(batch, n_head, seq_len, head_dim, dtype=cp.float32)
    k = cp.random.randn(batch, n_head, seq_len, head_dim, dtype=cp.float32)
    v = cp.random.randn(batch, n_head, seq_len, head_dim, dtype=cp.float32)

    y_cutile = cutile_causal_attention(q, k, v, n_head)
    y_cupy = cupy_causal_attention(q, k, v, n_head)

    print(f"Input Q shape: {q.shape}")
    print(f"Output shape: {y_cutile.shape}")
    print(f"Max diff: {cp.abs(y_cutile - y_cupy).max():.6f}")

    cp.testing.assert_allclose(y_cutile, y_cupy, atol=1e-3, rtol=1e-3)
    print("Causal attention test passed!")

    print("\n--- All Attention tests passed! ---")
