# SPDX-License-Identifier: Apache-2.0
"""
Linear layer kernel for cutile GPT.

Optimized MatMul with:
- 2D swizzle for L2 cache locality
- TF32 for tensor cores (float32)
- num_ctas and occupancy hints
- latency hints for memory access
- Fused bias addition
"""

import math
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Swizzle pattern for better L2 cache locality (from official sample)
GROUP_SIZE_M = 8


def swizzle_2d(M, N, tm, tn):
    """Get swizzled 2D block indices for better L2 locality."""
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def matmul_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    """
    Optimized matrix multiplication kernel: C = A @ B

    Args:
        A: Input matrix (M, K)
        B: Weight matrix (K, N)
        C: Output matrix (M, N)
        tm, tn, tk: Tile sizes
    """
    M = A.shape[0]
    N = B.shape[1]

    # 2D swizzle for L2 cache locality
    bid_m, bid_n = swizzle_2d(M, N, tm, tn)

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    # Accumulator in fp32 for precision
    acc = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # Use TF32 for tensor cores with float32 input
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    # K-loop with latency hints and TMA
    for k in range(num_tiles_k):
        a = ct.load(A, index=(bid_m, k), shape=(tm, tk),
                    padding_mode=zero_pad, latency=4, allow_tma=True).astype(dtype)
        b = ct.load(B, index=(k, bid_n), shape=(tk, tn),
                    padding_mode=zero_pad, latency=4, allow_tma=True).astype(dtype)
        acc = ct.mma(a, b, acc)

    ct.store(C, index=(bid_m, bid_n), tile=acc.astype(C.dtype))


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def matmul_bias_kernel(A, B, bias, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    """
    Fused matrix multiplication with bias: C = A @ B + bias

    Fusing bias avoids an extra kernel launch and memory round-trip.
    """
    M = A.shape[0]
    N = B.shape[1]

    # 2D swizzle for L2 cache locality
    bid_m, bid_n = swizzle_2d(M, N, tm, tn)

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    # Accumulator in fp32
    acc = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # Use TF32 for tensor cores
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    # K-loop with TMA
    for k in range(num_tiles_k):
        a = ct.load(A, index=(bid_m, k), shape=(tm, tk),
                    padding_mode=zero_pad, latency=4, allow_tma=True).astype(dtype)
        b = ct.load(B, index=(k, bid_n), shape=(tk, tn),
                    padding_mode=zero_pad, latency=4, allow_tma=True).astype(dtype)
        acc = ct.mma(a, b, acc)

    # Fused bias addition
    b_tile = ct.load(bias, index=(bid_n,), shape=(tn,),
                     padding_mode=zero_pad, latency=2, allow_tma=True)
    acc = acc + b_tile

    ct.store(C, index=(bid_m, bid_n), tile=acc.astype(C.dtype))


def cutile_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Linear transformation without bias: y = x @ weight.T

    Args:
        x: Input tensor (..., in_features)
        weight: Weight matrix (out_features, in_features)

    Returns:
        Output tensor (..., out_features)
    """
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("Tensors must be on CUDA device")

    original_shape = x.shape[:-1]
    in_features = x.shape[-1]
    out_features = weight.shape[0]

    # Reshape x to 2D: (batch * seq, in_features)
    x_2d = x.reshape(-1, in_features).contiguous()
    M = x_2d.shape[0]
    N = out_features

    # Transpose weight for x @ W^T
    weight_t = weight.t().contiguous()  # (in_features, out_features)

    # Output
    output = torch.empty(M, N, dtype=x.dtype, device=x.device)

    # Tile sizes - larger for fp16/bf16 (tensor cores), smaller for fp32
    if x.dtype in (torch.float16, torch.bfloat16):
        tm, tn, tk = 128, 128, 64  # Larger tiles for tensor cores
    else:
        tm, tn, tk = 32, 32, 32  # Use TF32 tensor cores

    grid_m = math.ceil(M / tm)
    grid_n = math.ceil(N / tn)
    grid = (grid_m * grid_n, 1, 1)

    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel,
              (x_2d, weight_t, output, tm, tn, tk))

    return output.reshape(*original_shape, out_features)


def cutile_linear_bias(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor
) -> torch.Tensor:
    """
    Fused linear transformation with bias: y = x @ weight.T + bias

    Uses fused kernel to avoid extra memory round-trip for bias.

    Args:
        x: Input tensor (..., in_features)
        weight: Weight matrix (out_features, in_features)
        bias: Bias vector (out_features,)

    Returns:
        Output tensor (..., out_features)
    """
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("Tensors must be on CUDA device")

    original_shape = x.shape[:-1]
    in_features = x.shape[-1]
    out_features = weight.shape[0]

    # Reshape x to 2D
    x_2d = x.reshape(-1, in_features).contiguous()
    M = x_2d.shape[0]
    N = out_features

    # Transpose weight
    weight_t = weight.t().contiguous()

    # Output
    output = torch.empty(M, N, dtype=x.dtype, device=x.device)

    # Tile sizes
    if x.dtype in (torch.float16, torch.bfloat16):
        tm, tn, tk = 128, 128, 64
    else:
        tm, tn, tk = 32, 32, 32

    grid_m = math.ceil(M / tm)
    grid_n = math.ceil(N / tn)
    grid = (grid_m * grid_n, 1, 1)

    # Use fused matmul+bias kernel
    ct.launch(torch.cuda.current_stream(), grid, matmul_bias_kernel,
              (x_2d, weight_t, bias, output, tm, tn, tk))

    return output.reshape(*original_shape, out_features)


# Reference PyTorch implementation
def torch_linear(x: torch.Tensor, weight: torch.Tensor,
                 bias: torch.Tensor = None) -> torch.Tensor:
    """PyTorch reference linear"""
    return torch.nn.functional.linear(x, weight, bias)


if __name__ == "__main__":
    print("--- Testing cutile Linear kernel ---")

    batch, seq, in_feat, out_feat = 2, 64, 48, 192

    x = torch.randn(batch, seq, in_feat, dtype=torch.float32, device='cuda')
    weight = torch.randn(out_feat, in_feat, dtype=torch.float32, device='cuda')
    bias = torch.randn(out_feat, dtype=torch.float32, device='cuda')

    # Test without bias
    y_cutile = cutile_linear(x, weight)
    y_torch = torch_linear(x, weight)

    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Output shape: {y_cutile.shape}")
    print(f"Without bias - Max diff: {(y_cutile - y_torch).abs().max().item():.6f}")
    torch.testing.assert_close(y_cutile, y_torch, atol=1e-4, rtol=1e-4)
    print("Linear (no bias) test passed!")

    # Test with bias
    y_cutile_bias = cutile_linear_bias(x, weight, bias)
    y_torch_bias = torch_linear(x, weight, bias)

    print(f"With bias - Max diff: {(y_cutile_bias - y_torch_bias).abs().max().item():.6f}")
    torch.testing.assert_close(y_cutile_bias, y_torch_bias, atol=1e-4, rtol=1e-4)
    print("Linear (with bias) test passed!")

    print("\n--- All Linear tests passed! ---")
