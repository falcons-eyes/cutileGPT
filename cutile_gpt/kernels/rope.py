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
                TILE_D: ConstInt):
    """
    Apply RoPE to one (TILE_M, head_dim/2) block of each half.

    X and Out are viewed as (batch, n_head, seq_len, 2, head_dim // 2) so the
    two halves sit on their own axis. Tile indices count tiles, not elements,
    so a padded tile on a combined head_dim axis would straddle the split -
    Phi's head_dim of 96 puts the boundary at 48, which is not a tile stride.
    Giving the halves an axis makes the split exact and lets TILE_D round up.

    Args:
        X: Input (batch, n_head, seq_len, 2, head_dim // 2)
        Cos: Cosine table (seq_len, head_dim // 2)
        Sin: Sine table (seq_len, head_dim // 2)
        Out: Output, same shape as X
        N_HEAD: Number of heads in X - query and key differ under GQA
        TILE_M: Tile size along the sequence
        TILE_D: Tile size across a half, rounded up to a power of two
    """
    bid_m = ct.bid(0)
    bid_bh = ct.bid(1)
    batch_idx = bid_bh // N_HEAD
    head_idx = bid_bh % N_HEAD

    x1 = ct.load(X, index=(batch_idx, head_idx, bid_m, 0, 0),
                 shape=(1, 1, TILE_M, 1, TILE_D),
                 padding_mode=PAD_ZERO, latency=4).reshape((TILE_M, TILE_D))
    x2 = ct.load(X, index=(batch_idx, head_idx, bid_m, 1, 0),
                 shape=(1, 1, TILE_M, 1, TILE_D),
                 padding_mode=PAD_ZERO, latency=4).reshape((TILE_M, TILE_D))

    cos = ct.load(Cos, index=(bid_m, 0), shape=(TILE_M, TILE_D),
                  padding_mode=PAD_ZERO, latency=2)
    sin = ct.load(Sin, index=(bid_m, 0), shape=(TILE_M, TILE_D),
                  padding_mode=PAD_ZERO, latency=2)

    # Rotate in fp32 regardless of the storage dtype. The angles are the same
    # for every head, so precision lost here shows up as a systematic position
    # error rather than noise.
    x1f = x1.astype(np.float32)
    x2f = x2.astype(np.float32)
    out1 = x1f * cos - x2f * sin
    out2 = x2f * cos + x1f * sin

    ct.store(Out, index=(batch_idx, head_idx, bid_m, 0, 0),
             tile=out1.reshape((1, 1, TILE_M, 1, TILE_D)).astype(Out.dtype))
    ct.store(Out, index=(batch_idx, head_idx, bid_m, 1, 0),
             tile=out2.reshape((1, 1, TILE_M, 1, TILE_D)).astype(Out.dtype))


def llama3_scale(inv_freq: cp.ndarray, factor: float, low_freq_factor: float,
                 high_freq_factor: float, original_context: int) -> cp.ndarray:
    """Stretch the low-frequency end of the RoPE spectrum, Llama 3 style.

    Llama 3.1 onward extend context by dividing the slow-turning components -
    the ones whose wavelength already exceeds the original training window - by
    `factor`, leaving the fast ones alone and interpolating between. Ignoring
    this does not fail loudly: the model still runs and still reads fluently,
    it just picks a different token perhaps one time in six.
    """
    wavelen = 2 * cp.pi / inv_freq
    low_wavelen = original_context / low_freq_factor
    high_wavelen = original_context / high_freq_factor

    scaled = inv_freq / factor
    smooth = ((original_context / wavelen - low_freq_factor)
              / (high_freq_factor - low_freq_factor))
    blended = (1 - smooth) * scaled + smooth * inv_freq

    out = cp.where(wavelen > low_wavelen, scaled, blended)
    return cp.where(wavelen < high_wavelen, inv_freq, out)


def rope_tables(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    offset: int = 0,
    dtype=cp.float32,
    scaling: dict | None = None,
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
        scaling: A checkpoint's `rope_scaling` block. Only `rope_type: llama3`
            is understood; anything else raises rather than being ignored,
            since a wrong frequency table produces plausible wrong tokens.

    Returns:
        (cos, sin), each (seq_len, head_dim // 2)
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    half = head_dim // 2
    # inv_freq[i] = 1 / theta^(2i/head_dim)
    inv_freq = 1.0 / (theta ** (cp.arange(0, half, dtype=cp.float64) * 2.0 / head_dim))

    if scaling:
        rope_type = scaling.get("rope_type") or scaling.get("type")
        if rope_type != "llama3":
            raise ValueError(
                f"rope_scaling type {rope_type!r} is not implemented; "
                "refusing to silently use unscaled frequencies"
            )
        inv_freq = llama3_scale(
            inv_freq,
            factor=float(scaling["factor"]),
            low_freq_factor=float(scaling["low_freq_factor"]),
            high_freq_factor=float(scaling["high_freq_factor"]),
            original_context=int(scaling["original_max_position_embeddings"]),
        )

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

    # cutile requires power-of-two tile dimensions, and neither axis obliges:
    # prompts are any length, and head_dim/2 is 48 for Phi. Loads pad with
    # zeros and stores clip at the array bound, so rounding both up costs a
    # few idle lanes and nothing else.
    tile_m = 64
    tile_d = 1 << (half - 1).bit_length()
    grid_m = (seq_len + tile_m - 1) // tile_m

    # Free reshape - the halves become their own axis so a tile index lands
    # exactly on the split.
    x_split = x.reshape(batch, n_head, seq_len, 2, half)
    out_split = out.reshape(batch, n_head, seq_len, 2, half)

    ct.launch(cp.cuda.get_current_stream(), (grid_m, batch * n_head, 1),
              rope_kernel, (x_split, cos, sin, out_split, n_head, tile_m, tile_d))

    return out


def cupy_rope(x: cp.ndarray, cos: cp.ndarray, sin: cp.ndarray) -> cp.ndarray:
    """CuPy reference RoPE."""
    half = x.shape[-1] // 2
    x32 = x.astype(cp.float32)
    x1, x2 = x32[..., :half], x32[..., half:]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    return cp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(x.dtype)
