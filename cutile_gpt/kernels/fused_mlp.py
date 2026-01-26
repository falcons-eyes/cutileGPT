# SPDX-License-Identifier: Apache-2.0
"""
Fused MLP kernel for cutile GPT.

Fuses: Linear(expand) -> GELU -> Linear(contract)
into a single kernel to minimize memory bandwidth.

Key optimization: intermediate activations stay in registers/shared memory
instead of going back to global memory.
"""

import math
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

# GELU constants
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
GELU_COEF = 0.044715


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def fused_mlp_kernel(
    X,           # Input: (M, N_in)
    W_fc,        # Expand weight: (N_hidden, N_in)
    B_fc,        # Expand bias: (N_hidden,)
    W_proj,      # Contract weight: (N_in, N_hidden)
    B_proj,      # Contract bias: (N_in,)
    Y,           # Output: (M, N_in)
    TM: ConstInt,
    TN: ConstInt,
    TK: ConstInt,
):
    """
    Fused MLP: Y = Linear(GELU(Linear(X)))

    This kernel fuses three operations:
    1. Linear expand: X @ W_fc^T + B_fc  (N_in -> N_hidden)
    2. GELU activation
    3. Linear contract: hidden @ W_proj^T + B_proj  (N_hidden -> N_in)

    The intermediate hidden activations (4x larger) never touch global memory.
    """
    M = X.shape[0]
    N_in = X.shape[1]
    N_hidden = W_fc.shape[0]

    # Block indices
    bid = ct.bid(0)
    num_n = ct.cdiv(N_in, TN)
    bid_m = bid // num_n
    bid_n = bid % num_n

    # ===== Phase 1: Compute hidden = X @ W_fc^T + B_fc =====
    # We compute this in tiles and immediately apply GELU

    # For the output projection, we need to accumulate over N_hidden
    # So we compute chunks of hidden, apply GELU, and accumulate into output

    # Initialize output accumulator
    acc = ct.full((TM, TN), 0, dtype=ct.float32)

    # Number of hidden tiles
    num_hidden_tiles = ct.cdiv(N_hidden, TK)

    for h in range(num_hidden_tiles):
        # Step 1: Compute a tile of the first linear (expand)
        # hidden_tile[m, k] = sum_j(X[m, j] * W_fc[k, j]) + B_fc[k]

        hidden_tile = ct.full((TM, TK), 0, dtype=ct.float32)

        # Accumulate over input dimension
        num_in_tiles = ct.cdiv(N_in, TN)
        for j in range(num_in_tiles):
            x_tile = ct.load(X, index=(bid_m, j), shape=(TM, TN),
                            padding_mode=ct.PaddingMode.ZERO)
            # W_fc: (N_hidden, N_in), we need W_fc[h*TK:(h+1)*TK, j*TN:(j+1)*TN]^T
            # which is (TN, TK) after transpose
            w_fc_tile = ct.load(W_fc, index=(h, j), shape=(TK, TN),
                               padding_mode=ct.PaddingMode.ZERO)
            # x_tile: (TM, TN) @ w_fc_tile^T: (TN, TK) -> (TM, TK)
            hidden_tile = ct.mma(x_tile, ct.transpose(w_fc_tile), hidden_tile)

        # Add bias
        b_fc_tile = ct.load(B_fc, index=(h,), shape=(TK,),
                           padding_mode=ct.PaddingMode.ZERO)
        hidden_tile = hidden_tile + b_fc_tile

        # Step 2: Apply GELU (in registers!)
        x3 = hidden_tile * hidden_tile * hidden_tile
        inner = SQRT_2_OVER_PI * (hidden_tile + GELU_COEF * x3)
        hidden_gelu = 0.5 * hidden_tile * (1.0 + ct.tanh(inner))

        # Step 3: Accumulate into output via second linear (contract)
        # Y[m, n] += sum_k(hidden_gelu[m, k] * W_proj[n, k])
        # W_proj: (N_in, N_hidden)
        w_proj_tile = ct.load(W_proj, index=(bid_n, h), shape=(TN, TK),
                             padding_mode=ct.PaddingMode.ZERO)
        # hidden_gelu: (TM, TK) @ w_proj_tile^T: (TK, TN) -> (TM, TN)
        acc = ct.mma(hidden_gelu, ct.transpose(w_proj_tile), acc)

    # Add bias for output projection
    b_proj_tile = ct.load(B_proj, index=(bid_n,), shape=(TN,),
                         padding_mode=ct.PaddingMode.ZERO)
    acc = acc + b_proj_tile

    # Store result
    ct.store(Y, index=(bid_m, bid_n), tile=acc.astype(Y.dtype))


def cutile_fused_mlp(
    x: torch.Tensor,
    w_fc: torch.Tensor,
    b_fc: torch.Tensor,
    w_proj: torch.Tensor,
    b_proj: torch.Tensor,
) -> torch.Tensor:
    """
    Fused MLP forward pass.

    Args:
        x: Input tensor (batch, seq_len, n_embd)
        w_fc: Expand weight (4*n_embd, n_embd)
        b_fc: Expand bias (4*n_embd,)
        w_proj: Contract weight (n_embd, 4*n_embd)
        b_proj: Contract bias (n_embd,)

    Returns:
        Output tensor (batch, seq_len, n_embd)
    """
    original_shape = x.shape
    n_embd = x.shape[-1]
    n_hidden = w_fc.shape[0]

    # Flatten to 2D
    x_2d = x.reshape(-1, n_embd).contiguous()
    M = x_2d.shape[0]

    # Output
    y = torch.empty(M, n_embd, dtype=x.dtype, device=x.device)

    # Tile sizes (powers of 2)
    TM = 32
    TN = 32
    TK = 32

    grid_m = math.ceil(M / TM)
    grid_n = math.ceil(n_embd / TN)
    grid = (grid_m * grid_n, 1, 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fused_mlp_kernel,
        (x_2d, w_fc, b_fc, w_proj, b_proj, y, TM, TN, TK)
    )

    return y.reshape(original_shape)


# Reference PyTorch implementation
def torch_mlp(x, w_fc, b_fc, w_proj, b_proj):
    """PyTorch reference MLP"""
    h = torch.nn.functional.linear(x, w_fc, b_fc)
    h = 0.5 * h * (1.0 + torch.tanh(SQRT_2_OVER_PI * (h + GELU_COEF * h**3)))
    return torch.nn.functional.linear(h, w_proj, b_proj)


if __name__ == "__main__":
    print("--- Testing Fused MLP kernel ---")

    batch, seq, n_embd = 2, 32, 64  # Use power of 2 for n_embd
    n_hidden = 4 * n_embd  # 256

    x = torch.randn(batch, seq, n_embd, dtype=torch.float32, device='cuda')
    w_fc = torch.randn(n_hidden, n_embd, dtype=torch.float32, device='cuda') * 0.02
    b_fc = torch.randn(n_hidden, dtype=torch.float32, device='cuda') * 0.02
    w_proj = torch.randn(n_embd, n_hidden, dtype=torch.float32, device='cuda') * 0.02
    b_proj = torch.randn(n_embd, dtype=torch.float32, device='cuda') * 0.02

    y_cutile = cutile_fused_mlp(x, w_fc, b_fc, w_proj, b_proj)
    y_torch = torch_mlp(x, w_fc, b_fc, w_proj, b_proj)

    diff = (y_cutile - y_torch).abs().max().item()
    print(f"Input: {x.shape}")
    print(f"Output: {y_cutile.shape}")
    print(f"Max diff: {diff:.6e}")

    if diff < 1e-3:
        print("Fused MLP: PASSED!")
    else:
        print("FAILED")
