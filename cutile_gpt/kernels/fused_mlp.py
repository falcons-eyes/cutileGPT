# SPDX-License-Identifier: Apache-2.0
"""Two-stage GELU MLP for cutile GPT.

The expand projection and GELU share one kernel; the contract projection is a
second kernel. A previous one-kernel implementation recomputed the complete
expand projection once per contract-output tile because independent tile
blocks cannot share register state. Avoiding that duplicated GEMM is far more
important than saving the remaining kernel boundary.
"""

import math

import cuda.tile as ct
import cupy as cp

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
GELU_COEF = 0.044715


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def gelu_projection_kernel(
    X,
    W,
    Bias,
    Hidden,
    TM: ConstInt,
    TN: ConstInt,
    TK: ConstInt,
):
    """Hidden = GELU(X @ W.T + Bias)."""
    bid = ct.bid(0)
    num_n = ct.cdiv(W.shape[0], TN)
    bid_m = bid // num_n
    bid_n = bid % num_n

    acc = ct.full((TM, TN), 0, dtype=ct.float32)
    dtype = ct.tfloat32 if X.dtype == ct.float32 else X.dtype
    for k in range(ct.cdiv(X.shape[1], TK)):
        x = ct.load(X, index=(bid_m, k), shape=(TM, TK),
                    padding_mode=PAD_ZERO, latency=4, allow_tma=True).astype(dtype)
        w = ct.load(W, index=(bid_n, k), shape=(TN, TK),
                    padding_mode=PAD_ZERO, latency=4, allow_tma=True).astype(dtype)
        acc = ct.mma(x, ct.transpose(w), acc)

    bias = ct.load(Bias, index=(bid_n,), shape=(TN,),
                   padding_mode=PAD_ZERO, latency=2, allow_tma=True)
    z = acc + bias
    z3 = z * z * z
    inner = SQRT_2_OVER_PI * (z + GELU_COEF * z3)
    activated = 0.5 * z * (1.0 + ct.tanh(inner))
    ct.store(Hidden, index=(bid_m, bid_n), tile=activated.astype(Hidden.dtype))


def cutile_fused_mlp(
    x: cp.ndarray,
    w_fc: cp.ndarray,
    b_fc: cp.ndarray,
    w_proj: cp.ndarray,
    b_proj: cp.ndarray,
    residual: cp.ndarray | None = None,
    w_proj_t: cp.ndarray | None = None,
) -> cp.ndarray:
    """Run expand+GELU and contract as two non-recomputing kernels."""
    from .linear import (
        cutile_linear_bias,
        cutile_linear_residual,
        linear_tile_sizes,
    )

    original_shape = x.shape
    n_embd = x.shape[-1]
    x_2d = cp.reshape(x, (-1, n_embd))
    if not x_2d.flags.c_contiguous:
        x_2d = cp.ascontiguousarray(x_2d)

    m = x_2d.shape[0]
    n_hidden = w_fc.shape[0]
    hidden = cp.empty((m, n_hidden), dtype=x.dtype)

    tm, tn, tk = linear_tile_sizes(x.dtype, m, n_hidden, n_embd)
    grid = (math.ceil(m / tm) * math.ceil(n_hidden / tn),)
    ct.launch(cp.cuda.get_current_stream(), grid, gelu_projection_kernel,
              (x_2d, w_fc, b_fc, hidden, tm, tn, tk))

    if residual is None:
        y = cutile_linear_bias(hidden, w_proj, b_proj)
        return cp.reshape(y, original_shape)
    return cutile_linear_residual(
        hidden, w_proj, residual, bias=b_proj, weight_t=w_proj_t
    )


def cupy_mlp(x, w_fc, b_fc, w_proj, b_proj):
    """CuPy reference MLP."""
    h = cp.matmul(x, w_fc.T) + b_fc
    h = 0.5 * h * (1.0 + cp.tanh(SQRT_2_OVER_PI * (h + GELU_COEF * h**3)))
    return cp.matmul(h, w_proj.T) + b_proj
