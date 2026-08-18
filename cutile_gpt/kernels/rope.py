# SPDX-License-Identifier: Apache-2.0
"""
Rotary position embedding (RoPE) kernel for cutile GPT.

GPT-2 adds a learned position vector to the token embedding once, at the
bottom of the stack. Every current open-weight model instead rotates Q and K
inside each attention layer by an angle that depends on absolute position, so
the QK product ends up depending only on relative position.

This follows the HuggingFace convention used by Llama, Qwen3, Gemma, and Muse
Glimmer, which splits head_dim in half rather than pairing adjacent elements:

    rotate_half(x) = cat(-x2, x1)
    out = x * cos + rotate_half(x) * sin

Written per half, which is what the kernel actually computes:

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
"""

import cuda.tile as ct
import cupy as cp
import numpy as np

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def rope_kernel(X, Cos, Sin, Out, N_HEAD: ConstInt, TILE_M: ConstInt,
                HALF_DIM: ConstInt):
    """
    Apply RoPE to one (TILE_M, head_dim) block.

    Args:
        X: Input (batch, n_head, seq_len, head_dim)
        Cos: Cosine table (seq_len, head_dim // 2)
        Sin: Sine table (seq_len, head_dim // 2)
        Out: Output, same shape as X
        N_HEAD: Number of heads in X - query and key differ under GQA
        TILE_M: Tile size along the sequence
        HALF_DIM: head_dim // 2
    """
    bid_m = ct.bid(0)
    bid_bh = ct.bid(1)
    batch_idx = bid_bh // N_HEAD
    head_idx = bid_bh % N_HEAD

    # The two halves are just the two column tiles of the head, so no slicing
    # is needed - each is a load at its own column index.
    x1 = ct.load(X, index=(batch_idx, head_idx, bid_m, 0),
                 shape=(1, 1, TILE_M, HALF_DIM),
                 padding_mode=PAD_ZERO, latency=4).reshape((TILE_M, HALF_DIM))
    x2 = ct.load(X, index=(batch_idx, head_idx, bid_m, 1),
                 shape=(1, 1, TILE_M, HALF_DIM),
                 padding_mode=PAD_ZERO, latency=4).reshape((TILE_M, HALF_DIM))

    cos = ct.load(Cos, index=(bid_m, 0), shape=(TILE_M, HALF_DIM),
                  padding_mode=PAD_ZERO, latency=2)
    sin = ct.load(Sin, index=(bid_m, 0), shape=(TILE_M, HALF_DIM),
                  padding_mode=PAD_ZERO, latency=2)

    # Rotate in fp32 regardless of the storage dtype. The angles are the same
    # for every head, so precision lost here shows up as a systematic position
    # error rather than noise.
    x1f = x1.astype(np.float32)
    x2f = x2.astype(np.float32)
    out1 = x1f * cos - x2f * sin
    out2 = x2f * cos + x1f * sin

    ct.store(Out, index=(batch_idx, head_idx, bid_m, 0),
             tile=out1.reshape((1, 1, TILE_M, HALF_DIM)).astype(Out.dtype))
    ct.store(Out, index=(batch_idx, head_idx, bid_m, 1),
             tile=out2.reshape((1, 1, TILE_M, HALF_DIM)).astype(Out.dtype))


def rope_tables(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    offset: int = 0,
    dtype=cp.float32,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Build the cos/sin tables for a run of positions.

    Args:
        seq_len: Number of positions to generate
        head_dim: Attention head dimension
        theta: RoPE base. 10000 is the original value; long-context models
            raise it - Muse Glimmer uses 500000, and a larger base stretches
            the wavelengths so distant positions stay distinguishable.
        offset: Absolute index of the first position. Non-zero when decoding
            with a cache, where the new token is not at position 0.
        dtype: dtype of the returned tables

    Returns:
        (cos, sin), each (seq_len, head_dim // 2)
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    half = head_dim // 2
    # inv_freq[i] = 1 / theta^(2i/head_dim)
    inv_freq = 1.0 / (theta ** (cp.arange(0, half, dtype=cp.float64) * 2.0 / head_dim))
    pos = cp.arange(offset, offset + seq_len, dtype=cp.float64)
    freqs = pos[:, None] * inv_freq[None, :]
    return (cp.ascontiguousarray(cp.cos(freqs).astype(dtype)),
            cp.ascontiguousarray(cp.sin(freqs).astype(dtype)))


def cutile_rope(
    x: cp.ndarray,
    cos: cp.ndarray,
    sin: cp.ndarray,
) -> cp.ndarray:
    """
    Apply rotary position embedding to Q or K.

    Call it separately for queries and keys; under GQA they have different
    head counts but share the same tables.

    Args:
        x: (batch, n_head, seq_len, head_dim)
        cos: (seq_len, head_dim // 2), from `rope_tables`
        sin: (seq_len, head_dim // 2)

    Returns:
        Rotated tensor with the same shape and dtype as x
    """
    if not isinstance(x, cp.ndarray):
        raise ValueError("Input tensor must be a CuPy array on CUDA device")
    if x.ndim != 4:
        raise ValueError(f"expected (batch, n_head, seq_len, head_dim), got {x.shape}")

    batch, n_head, seq_len, head_dim = x.shape
    half = head_dim // 2

    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    if cos.shape != (seq_len, half) or sin.shape != (seq_len, half):
        raise ValueError(
            f"cos/sin must be ({seq_len}, {half}), got {cos.shape} and {sin.shape}"
        )

    if not x.flags.c_contiguous:
        x = cp.ascontiguousarray(x)

    out = cp.empty_like(x)

    tile_m = min(64, seq_len)
    grid_m = (seq_len + tile_m - 1) // tile_m

    ct.launch(cp.cuda.get_current_stream(), (grid_m, batch * n_head, 1),
              rope_kernel, (x, cos, sin, out, n_head, tile_m, half))

    return out


def cupy_rope(x: cp.ndarray, cos: cp.ndarray, sin: cp.ndarray) -> cp.ndarray:
    """CuPy reference RoPE."""
    half = x.shape[-1] // 2
    x32 = x.astype(cp.float32)
    x1, x2 = x32[..., :half], x32[..., half:]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    return cp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(x.dtype)
