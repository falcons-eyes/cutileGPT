# SPDX-License-Identifier: Apache-2.0
"""
LayerNorm kernel for cutile GPT.

cutile requires tile sizes to be powers of 2.
We pad dimensions and slice back.
"""

import math
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def layer_norm_kernel(X, W, B, Y, eps, N: ConstInt, TILE_N: ConstInt):
    """
    Optimized Forward LayerNorm kernel using Welford's algorithm.

    Computes mean and variance in single pass (2 passes total vs 3).

    Args:
        X: Input tensor (M, N_padded)
        W: Weight tensor (N_padded,)
        B: Bias tensor (N_padded,)
        Y: Output tensor (M, N_padded)
        eps: Epsilon for numerical stability
        N: Actual (unpadded) normalized dimension
        TILE_N: Tile size (power of 2)
    """
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_N))

    # Pass 1: Compute mean and variance together using sum and sum of squares
    sum_acc = ct.full((1, TILE_N), 0, dtype=ct.float32)
    sum_sq_acc = ct.full((1, TILE_N), 0, dtype=ct.float32)

    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N),
                    padding_mode=PAD_ZERO, latency=4, allow_tma=True)
        # Mask out padded values
        col_idx = j * TILE_N + ct.arange(TILE_N, dtype=ct.int32)
        mask = col_idx < N
        tx_masked = ct.where(mask, tx, 0)
        sum_acc += tx_masked
        sum_sq_acc += tx_masked * tx_masked

    # Compute mean and variance: var = E[X^2] - E[X]^2
    total_sum = ct.sum(sum_acc, axis=1)
    total_sum_sq = ct.sum(sum_sq_acc, axis=1)
    mean = total_sum / N
    var = total_sum_sq / N - mean * mean
    rstd = 1 / ct.sqrt(var + eps)

    # Pass 2: Normalize and apply affine transform
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N),
                    padding_mode=PAD_ZERO, latency=4, allow_tma=True)
        tw = ct.load(W, index=(j,), shape=(TILE_N,),
                    padding_mode=PAD_ZERO, latency=2, allow_tma=True)
        tb = ct.load(B, index=(j,), shape=(TILE_N,),
                    padding_mode=PAD_ZERO, latency=2, allow_tma=True)
        ty = (tx - mean) * rstd
        ty = ty * tw + tb
        ct.store(Y, index=(bid_m, j), tile=ty.astype(Y.dtype))


def cutile_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Apply Layer Normalization using cutile kernel.

    Handles non-power-of-2 dimensions by padding.

    Args:
        x: Input tensor (..., normalized_shape)
        weight: Scale parameter (normalized_shape,)
        bias: Bias parameter (normalized_shape,)
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor with same shape as input
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")

    original_shape = x.shape
    n_embd = x.shape[-1]

    # Pad to power of 2
    n_embd_padded = next_power_of_2(n_embd)
    TILE_N = min(1024, n_embd_padded)
    # Make sure TILE_N is power of 2 and divides n_embd_padded
    while n_embd_padded % TILE_N != 0:
        TILE_N //= 2

    needs_padding = n_embd_padded != n_embd

    # Flatten to 2D
    x_2d = x.reshape(-1, n_embd)
    M = x_2d.shape[0]

    if needs_padding:
        x_padded = torch.zeros(M, n_embd_padded, dtype=x.dtype, device=x.device)
        x_padded[:, :n_embd] = x_2d
        weight_padded = torch.zeros(n_embd_padded, dtype=weight.dtype, device=weight.device)
        weight_padded[:n_embd] = weight
        bias_padded = torch.zeros(n_embd_padded, dtype=bias.dtype, device=bias.device)
        bias_padded[:n_embd] = bias
    else:
        x_padded = x_2d
        weight_padded = weight
        bias_padded = bias

    y_padded = torch.empty_like(x_padded)

    ct.launch(torch.cuda.current_stream(), (M,), layer_norm_kernel,
              (x_padded, weight_padded, bias_padded, y_padded, eps, n_embd, TILE_N))

    # Slice back
    y = y_padded[:, :n_embd]
    return y.reshape(original_shape)


# Reference PyTorch implementation
def torch_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """PyTorch reference LayerNorm"""
    return torch.nn.functional.layer_norm(x, weight.shape, weight, bias, eps)


if __name__ == "__main__":
    print("--- Testing cutile LayerNorm kernel ---")

    batch, seq, n_embd = 2, 64, 48

    x = torch.randn(batch, seq, n_embd, dtype=torch.float32, device='cuda')
    weight = torch.randn(n_embd, dtype=torch.float32, device='cuda')
    bias = torch.randn(n_embd, dtype=torch.float32, device='cuda')
    eps = 1e-5

    y_cutile = cutile_layer_norm(x, weight, bias, eps)
    y_torch = torch_layer_norm(x, weight, bias, eps)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_cutile.shape}")
    print(f"Max diff: {(y_cutile - y_torch).abs().max().item():.6f}")

    torch.testing.assert_close(y_cutile, y_torch, atol=1e-4, rtol=1e-4)
    print("LayerNorm test passed!")

    print("\n--- All LayerNorm tests passed! ---")
