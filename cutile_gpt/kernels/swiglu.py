# SPDX-License-Identifier: Apache-2.0
"""Two-stage SwiGLU MLP for modern decoder-only transformers.

Gate/up projection and the SiLU product share one kernel. The down projection
is deliberately separate: fusing it into every output tile would recompute the
entire gate/up projection because independent tile blocks cannot share their
register-resident hidden values.
"""

import math

import cuda.tile as ct
import cupy as cp

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def swiglu_gate_up_kernel(
    X,
    W_gate,
    W_up,
    Hidden,
    TM: ConstInt,
    TN: ConstInt,
    TK: ConstInt,
):
    """Hidden = silu(X @ W_gate.T) * (X @ W_up.T)."""
    bid = ct.bid(0)
    num_n = ct.cdiv(W_gate.shape[0], TN)
    bid_m = bid // num_n
    bid_n = bid % num_n

    gate = ct.full((TM, TN), 0, dtype=ct.float32)
    up = ct.full((TM, TN), 0, dtype=ct.float32)
    dtype = ct.tfloat32 if X.dtype == ct.float32 else X.dtype

    for k in range(ct.cdiv(X.shape[1], TK)):
        x = ct.load(X, index=(bid_m, k), shape=(TM, TK),
                    padding_mode=PAD_ZERO, latency=4, allow_tma=True).astype(dtype)
        wg = ct.load(W_gate, index=(bid_n, k), shape=(TN, TK),
                     padding_mode=PAD_ZERO, latency=4, allow_tma=True).astype(dtype)
        wu = ct.load(W_up, index=(bid_n, k), shape=(TN, TK),
                     padding_mode=PAD_ZERO, latency=4, allow_tma=True).astype(dtype)
        gate = ct.mma(x, ct.transpose(wg), gate)
        up = ct.mma(x, ct.transpose(wu), up)

    hidden = gate / (1.0 + ct.exp(-gate)) * up
    ct.store(Hidden, index=(bid_m, bid_n), tile=hidden.astype(Hidden.dtype))


def cutile_swiglu_mlp(
    x: cp.ndarray,
    w_gate: cp.ndarray,
    w_up: cp.ndarray,
    w_down: cp.ndarray,
    residual: cp.ndarray | None = None,
) -> cp.ndarray:
    """Run fused gate/up+SiLU followed by one down-projection kernel."""
    from .linear import cutile_linear, cutile_linear_residual, linear_tile_sizes

    if not isinstance(x, cp.ndarray):
        raise ValueError("Input tensor must be a CuPy array on CUDA device")
    if w_gate.shape != w_up.shape:
        raise ValueError(
            f"gate and up projections must match, got {w_gate.shape} and {w_up.shape}"
        )

    original_shape = x.shape
    n_embd = x.shape[-1]
    n_hidden = w_gate.shape[0]
    if w_down.shape != (n_embd, n_hidden):
        raise ValueError(
            f"down projection must be ({n_embd}, {n_hidden}), got {w_down.shape}"
        )

    x_2d = cp.reshape(x, (-1, n_embd))
    if not x_2d.flags.c_contiguous:
        x_2d = cp.ascontiguousarray(x_2d)
    m = x_2d.shape[0]
    hidden = cp.empty((m, n_hidden), dtype=x.dtype)

    tm, tn, tk = linear_tile_sizes(x.dtype, m, n_hidden, n_embd)
    # The gate/up kernel issues two MMAs per input tile. On the decode path a
    # shallower K tile overlaps those two weight streams better than the
    # single-projection default selected by linear_tile_sizes.
    if m <= 16 and x.dtype.itemsize == 2:
        tm, tn, tk = 16, 128, 64
    grid = (math.ceil(m / tm) * math.ceil(n_hidden / tn),)
    ct.launch(cp.cuda.get_current_stream(), grid, swiglu_gate_up_kernel,
              (x_2d, w_gate, w_up, hidden, tm, tn, tk))

    if residual is None:
        y = cutile_linear(hidden, w_down)
        return cp.reshape(y, original_shape)
    return cutile_linear_residual(hidden, w_down, residual)


def cupy_swiglu_mlp(
    x: cp.ndarray,
    w_gate: cp.ndarray,
    w_up: cp.ndarray,
    w_down: cp.ndarray,
) -> cp.ndarray:
    """CuPy reference SwiGLU MLP."""
    x32 = x.astype(cp.float32).reshape(-1, x.shape[-1])
    gate = x32 @ w_gate.astype(cp.float32).T
    up = x32 @ w_up.astype(cp.float32).T
    hidden = gate / (1.0 + cp.exp(-gate)) * up
    out = hidden @ w_down.astype(cp.float32).T
    return out.reshape(x.shape).astype(x.dtype)
