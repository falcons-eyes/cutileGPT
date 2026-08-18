# SPDX-License-Identifier: Apache-2.0
"""
Fused SwiGLU MLP kernel for cutile GPT.

Qwen3, Gemma, Llama, and Muse Glimmer all replace GPT-2's
`down(gelu(up(x)))` with a gated variant:

    hidden = silu(x @ W_gate^T) * (x @ W_up^T)
    y      = hidden @ W_down^T

Two expand projections instead of one, multiplied elementwise, and no biases.
The gate and up projections share the same X tiles, so both are computed inside
one pass over the input and the intermediate hidden activations - which are
`intermediate_size` wide, 3x n_embd in Muse Glimmer - never reach global memory.
"""

import cuda.tile as ct
import cupy as cp

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def swiglu_mlp_kernel(
    X,        # Input: (M, N_in)
    W_gate,   # Gate weight: (N_hidden, N_in)
    W_up,     # Up weight: (N_hidden, N_in)
    W_down,   # Down weight: (N_in, N_hidden)
    Y,        # Output: (M, N_in)
    TM: ConstInt,
    TN: ConstInt,
    TK: ConstInt,
):
    """
    Fused SwiGLU MLP: Y = (silu(X @ W_gate^T) * (X @ W_up^T)) @ W_down^T
    """
    N_in = X.shape[1]
    N_hidden = W_gate.shape[0]

    bid = ct.bid(0)
    num_n = ct.cdiv(N_in, TN)
    bid_m = bid // num_n
    bid_n = bid % num_n

    acc = ct.full((TM, TN), 0, dtype=ct.float32)
    num_hidden_tiles = ct.cdiv(N_hidden, TK)
    num_in_tiles = ct.cdiv(N_in, TN)

    for h in range(num_hidden_tiles):
        gate_tile = ct.full((TM, TK), 0, dtype=ct.float32)
        up_tile = ct.full((TM, TK), 0, dtype=ct.float32)

        # One pass over X feeding both projections - the X tile is loaded once
        # and used twice, which is the reason to fuse the gate and up matmuls
        # rather than run them as two separate kernels.
        for j in range(num_in_tiles):
            x_tile = ct.load(X, index=(bid_m, j), shape=(TM, TN),
                             padding_mode=PAD_ZERO)
            w_gate_tile = ct.load(W_gate, index=(h, j), shape=(TK, TN),
                                  padding_mode=PAD_ZERO)
            w_up_tile = ct.load(W_up, index=(h, j), shape=(TK, TN),
                                padding_mode=PAD_ZERO)
            gate_tile = ct.mma(x_tile, ct.transpose(w_gate_tile), gate_tile)
            up_tile = ct.mma(x_tile, ct.transpose(w_up_tile), up_tile)

        # SwiGLU in registers: silu(gate) * up, where silu(z) = z * sigmoid(z).
        # Written as z / (1 + exp(-z)) so it stays one exp and one divide.
        hidden = gate_tile / (1.0 + ct.exp(-gate_tile)) * up_tile

        w_down_tile = ct.load(W_down, index=(bid_n, h), shape=(TN, TK),
                              padding_mode=PAD_ZERO)
        acc = ct.mma(hidden, ct.transpose(w_down_tile), acc)

    ct.store(Y, index=(bid_m, bid_n), tile=acc.astype(Y.dtype))


def cutile_swiglu_mlp(
    x: cp.ndarray,
    w_gate: cp.ndarray,
    w_up: cp.ndarray,
    w_down: cp.ndarray,
) -> cp.ndarray:
    """
    Fused SwiGLU MLP forward pass.

    Weight shapes follow the HuggingFace layout for these models, so
    `gate_proj.weight`, `up_proj.weight`, and `down_proj.weight` can be passed
    through unchanged.

    Args:
        x: Input tensor (batch, seq_len, n_embd)
        w_gate: Gate projection (intermediate_size, n_embd)
        w_up: Up projection (intermediate_size, n_embd)
        w_down: Down projection (n_embd, intermediate_size)

    Returns:
        Output tensor (batch, seq_len, n_embd)
    """
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
    M = x_2d.shape[0]

    y = cp.empty((M, n_embd), dtype=x.dtype)

    TM, TN, TK = 64, 64, 64
    grid = ((M + TM - 1) // TM) * ((n_embd + TN - 1) // TN)

    ct.launch(cp.cuda.get_current_stream(), (grid,), swiglu_mlp_kernel,
              (x_2d, w_gate, w_up, w_down, y, TM, TN, TK))

    return cp.reshape(y, original_shape)


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
