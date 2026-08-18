# SPDX-License-Identifier: Apache-2.0
"""
RMSNorm kernel for cutile GPT.

Every current open-weight model - Qwen3, Gemma, Llama, Muse Glimmer - normalizes
with RMSNorm rather than LayerNorm. It drops the mean subtraction and the bias,
leaving y = x * rsqrt(mean(x^2) + eps) * w, so it needs one accumulator instead
of two and one fewer tensor.

cutile requires tile sizes to be powers of 2, so dimensions are padded and
sliced back the same way layernorm.py does it.
"""

import cuda.tile as ct
import cupy as cp

from .layernorm import next_power_of_2

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]
PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def rms_norm_kernel(X, W, Y, eps, N: ConstInt, TILE_N: ConstInt,
                    UNIT_OFFSET: ConstBool):
    """
    Forward RMSNorm.

    Args:
        X: Input tensor (M, N_padded)
        W: Scale tensor (N_padded,)
        Y: Output tensor (M, N_padded)
        eps: Epsilon inside the square root
        N: Actual (unpadded) normalized dimension
        TILE_N: Tile size (power of 2)
        UNIT_OFFSET: Scale by (1 + W) instead of W, the Gemma convention
    """
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_N))

    # Pass 1: sum of squares only - no mean, which is the whole difference
    # from LayerNorm.
    sum_sq_acc = ct.full((1, TILE_N), 0, dtype=ct.float32)

    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N),
                     padding_mode=PAD_ZERO, latency=4, allow_tma=True)
        # Padding lanes must not enter the sum; they are zeroed, and zero is
        # not neutral once we divide by the true N.
        col_idx = j * TILE_N + ct.arange(TILE_N, dtype=ct.int32)
        tx_masked = ct.where(col_idx < N, tx, 0)
        sum_sq_acc += tx_masked * tx_masked

    mean_sq = ct.sum(sum_sq_acc, axis=1) / N
    rstd = 1 / ct.sqrt(mean_sq + eps)

    # Pass 2: normalize and scale
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N),
                     padding_mode=PAD_ZERO, latency=4, allow_tma=True)
        tw = ct.load(W, index=(j,), shape=(TILE_N,),
                     padding_mode=PAD_ZERO, latency=2, allow_tma=True)
        if UNIT_OFFSET:
            tw = tw + 1.0
        ty = tx * rstd * tw
        ct.store(Y, index=(bid_m, j), tile=ty.astype(Y.dtype))


def cutile_rms_norm(
    x: cp.ndarray,
    weight: cp.ndarray,
    eps: float = 1e-6,
    unit_offset: bool = False,
) -> cp.ndarray:
    """
    Apply RMSNorm using a cutile kernel.

    Args:
        x: Input tensor (..., normalized_shape)
        weight: Scale parameter (normalized_shape,)
        eps: Epsilon inside the square root. Note the default is 1e-6, matching
            what Qwen/Llama configs ship; LayerNorm here defaults to 1e-5.
        unit_offset: Scale by (1 + weight). Gemma stores its RMSNorm weights
            centered on zero and applies this offset at runtime; Qwen, Llama,
            and Muse Glimmer do not.

    Returns:
        Normalized tensor with the same shape as the input
    """
    if not isinstance(x, cp.ndarray):
        raise ValueError("Input tensor must be a CuPy array on CUDA device")

    original_shape = x.shape
    n_embd = x.shape[-1]

    n_embd_padded = next_power_of_2(n_embd)
    TILE_N = min(1024, n_embd_padded)
    while n_embd_padded % TILE_N != 0:
        TILE_N //= 2

    x_2d = cp.reshape(x, (-1, n_embd))
    M = x_2d.shape[0]
    y = cp.empty_like(x_2d)

    ct.launch(cp.cuda.get_current_stream(), (M,), rms_norm_kernel,
              (x_2d, weight, y, eps, n_embd, TILE_N,
               unit_offset))

    return cp.reshape(y, original_shape)


def cupy_rms_norm(
    x: cp.ndarray,
    weight: cp.ndarray,
    eps: float = 1e-6,
    unit_offset: bool = False,
) -> cp.ndarray:
    """CuPy reference RMSNorm."""
    x32 = x.astype(cp.float32)
    rstd = 1.0 / cp.sqrt(cp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    w = weight.astype(cp.float32)
    if unit_offset:
        w = w + 1.0
    return (x32 * rstd * w).astype(x.dtype)
