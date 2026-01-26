# SPDX-License-Identifier: Apache-2.0
"""
Optimized Linear layer kernel for cutile GPT.

Key optimizations:
- Adaptive tile sizes based on matrix dimensions
- Removed redundant transpose operations via caching
- Improved swizzle pattern for small workloads
- Better occupancy hints for GB10 GPU
"""

import math
import cupy as cp
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Adaptive swizzle group size
def get_swizzle_group_size(M, N):
    """Determine optimal swizzle group size based on workload."""
    # For small workloads, use smaller group size to reduce idle threads
    total_tiles = (M // 32) * (N // 32)
    if total_tiles < 64:
        return 2
    elif total_tiles < 256:
        return 4
    else:
        return 8


def swizzle_2d(M, N, tm, tn, group_size):
    """Get swizzled 2D block indices for better L2 locality."""
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = group_size * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * group_size
    group_size_m = min(num_bid_m - first_bid_m, group_size)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def matmul_kernel_opt(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt, group_size: ConstInt):
    """
    Optimized matrix multiplication kernel: C = A @ B

    Args:
        A: Input matrix (M, K)
        B: Weight matrix (K, N) - already transposed if needed
        C: Output matrix (M, N)
        tm, tn, tk: Tile sizes
        group_size: Swizzle group size
    """
    M = A.shape[0]
    N = B.shape[1]

    # Adaptive 2D swizzle
    bid_m, bid_n = swizzle_2d(M, N, tm, tn, group_size)

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    # Accumulator in fp32
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
def matmul_bias_kernel_opt(A, B, bias, C, tm: ConstInt, tn: ConstInt, tk: ConstInt, group_size: ConstInt):
    """
    Fused matrix multiplication with bias: C = A @ B + bias

    Args:
        A: Input matrix (M, K)
        B: Weight matrix (K, N) - already transposed if needed
        bias: Bias vector (N,)
        C: Output matrix (M, N)
        tm, tn, tk: Tile sizes
        group_size: Swizzle group size
    """
    M = A.shape[0]
    N = B.shape[1]

    # Adaptive 2D swizzle
    bid_m, bid_n = swizzle_2d(M, N, tm, tn, group_size)

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


def get_optimal_tile_sizes(M, N, K, dtype):
    """
    Determine optimal tile sizes based on matrix dimensions and dtype.

    Strategy:
    - Large matrices: Use large tiles for better tensor core utilization
    - Medium/small matrices: Use 32x32 for balance
    - fp16/bf16: Larger tiles benefit more from tensor cores
    - fp32: TF32 tensor cores with conservative tile sizes
    """
    is_fp16 = dtype in (cp.float16, cp.dtype('float16'))

    # Estimate total work (in tiles of base size 32)
    base_tiles = (M // 32) * (N // 32)

    if is_fp16:
        # fp16 benefits from larger tiles
        if base_tiles > 512:
            return 128, 128, 64
        elif base_tiles > 128:
            return 64, 64, 32
        else:
            return 32, 32, 32
    else:
        # fp32 with TF32 tensor cores
        # Keep minimum 32x32 for numerical stability
        if base_tiles > 512:
            return 64, 64, 32
        else:
            return 32, 32, 32


def cutile_linear_opt(x: cp.ndarray, weight_t: cp.ndarray) -> cp.ndarray:
    """
    Optimized linear transformation without bias: y = x @ weight.T

    This version expects weight_t to be ALREADY TRANSPOSED.
    This avoids redundant transpose operations in the forward pass.

    Args:
        x: Input tensor (..., in_features)
        weight_t: Pre-transposed weight matrix (in_features, out_features)

    Returns:
        Output tensor (..., out_features)
    """
    if not isinstance(x, cp.ndarray) or not isinstance(weight_t, cp.ndarray):
        raise ValueError("Tensors must be CuPy arrays on CUDA device")

    original_shape = x.shape[:-1]
    in_features = x.shape[-1]
    out_features = weight_t.shape[1]

    # Reshape x to 2D: (batch * seq, in_features)
    x_2d = cp.reshape(x, (-1, in_features))
    if not x_2d.flags.c_contiguous:
        x_2d = cp.ascontiguousarray(x_2d)
    M = x_2d.shape[0]
    N = out_features
    K = in_features

    # Output
    output = cp.empty((M, N), dtype=x.dtype)

    # Adaptive tile sizes based on workload
    tm, tn, tk = get_optimal_tile_sizes(M, N, K, x.dtype)

    # Adaptive swizzle group size
    group_size = get_swizzle_group_size(M, N)

    grid_m = math.ceil(M / tm)
    grid_n = math.ceil(N / tn)
    grid = (grid_m * grid_n, 1, 1)

    ct.launch(cp.cuda.get_current_stream(), grid, matmul_kernel_opt,
              (x_2d, weight_t, output, tm, tn, tk, group_size))

    return cp.reshape(output, (*original_shape, out_features))


def cutile_linear_bias_opt(
    x: cp.ndarray,
    weight_t: cp.ndarray,
    bias: cp.ndarray
) -> cp.ndarray:
    """
    Fused linear transformation with bias: y = x @ weight.T + bias

    This version expects weight_t to be ALREADY TRANSPOSED.

    Args:
        x: Input tensor (..., in_features)
        weight_t: Pre-transposed weight matrix (in_features, out_features)
        bias: Bias vector (out_features,)

    Returns:
        Output tensor (..., out_features)
    """
    if not isinstance(x, cp.ndarray) or not isinstance(weight_t, cp.ndarray):
        raise ValueError("Tensors must be CuPy arrays on CUDA device")

    original_shape = x.shape[:-1]
    in_features = x.shape[-1]
    out_features = weight_t.shape[1]

    # Reshape x to 2D
    x_2d = cp.reshape(x, (-1, in_features))
    if not x_2d.flags.c_contiguous:
        x_2d = cp.ascontiguousarray(x_2d)
    M = x_2d.shape[0]
    N = out_features
    K = in_features

    # Output
    output = cp.empty((M, N), dtype=x.dtype)

    # Adaptive tile sizes
    tm, tn, tk = get_optimal_tile_sizes(M, N, K, x.dtype)

    # Adaptive swizzle group size
    group_size = get_swizzle_group_size(M, N)

    grid_m = math.ceil(M / tm)
    grid_n = math.ceil(N / tn)
    grid = (grid_m * grid_n, 1, 1)

    # Use fused matmul+bias kernel
    ct.launch(cp.cuda.get_current_stream(), grid, matmul_bias_kernel_opt,
              (x_2d, weight_t, bias, output, tm, tn, tk, group_size))

    return cp.reshape(output, (*original_shape, out_features))


def precompute_weight_transposes(weights_dict):
    """
    Precompute and cache all weight transposes.

    This is called once during model initialization to avoid
    redundant transpose operations during forward pass.

    Args:
        weights_dict: Dictionary containing model weights

    Returns:
        Dictionary with transposed weights (adds '_t' suffix keys)
    """
    transposed = {}

    for key, weight in weights_dict.items():
        if 'weight' in key and weight.ndim == 2:
            # Transpose and ensure contiguous
            weight_t = cp.transpose(weight)
            if not weight_t.flags.c_contiguous:
                weight_t = cp.ascontiguousarray(weight_t)

            # Store with '_t' suffix
            transposed[key + '_t'] = weight_t

    return transposed


# Reference implementation using CuPy
def cupy_linear(x: cp.ndarray, weight: cp.ndarray,
                bias: cp.ndarray = None) -> cp.ndarray:
    """CuPy reference linear: y = x @ weight.T + bias"""
    y = cp.matmul(x, weight.T)
    if bias is not None:
        y = y + bias
    return y


if __name__ == "__main__":
    print("--- Testing Optimized Linear kernel ---")

    batch, seq, in_feat, out_feat = 8, 128, 128, 512

    x = cp.random.randn(batch, seq, in_feat, dtype=cp.float32)
    weight = cp.random.randn(out_feat, in_feat, dtype=cp.float32)
    bias = cp.random.randn(out_feat, dtype=cp.float32)

    # Precompute transpose
    weight_t = cp.transpose(weight)
    if not weight_t.flags.c_contiguous:
        weight_t = cp.ascontiguousarray(weight_t)

    # Test without bias
    y_cutile = cutile_linear_opt(x, weight_t)
    y_cupy = cupy_linear(x, weight)

    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Output shape: {y_cutile.shape}")
    print(f"Without bias - Max diff: {cp.abs(y_cutile - y_cupy).max():.6f}")
    cp.testing.assert_allclose(y_cutile, y_cupy, atol=1e-4, rtol=1e-4)
    print("✓ Linear (no bias) test passed!")

    # Test with bias
    y_cutile_bias = cutile_linear_bias_opt(x, weight_t, bias)
    y_cupy_bias = cupy_linear(x, weight, bias)

    print(f"With bias - Max diff: {cp.abs(y_cutile_bias - y_cupy_bias).max():.6f}")
    cp.testing.assert_allclose(y_cutile_bias, y_cupy_bias, atol=1e-4, rtol=1e-4)
    print("✓ Linear (with bias) test passed!")

    print("\n--- Benchmarking ---")
    import time

    # Warmup
    for _ in range(10):
        _ = cutile_linear_bias_opt(x, weight_t, bias)
    cp.cuda.Stream.null.synchronize()

    # Benchmark optimized
    start = time.time()
    for _ in range(100):
        _ = cutile_linear_bias_opt(x, weight_t, bias)
    cp.cuda.Stream.null.synchronize()
    opt_time = (time.time() - start) / 100 * 1000

    # Benchmark cupy
    start = time.time()
    for _ in range(100):
        _ = cupy_linear(x, weight, bias)
    cp.cuda.Stream.null.synchronize()
    cupy_time = (time.time() - start) / 100 * 1000

    print(f"Optimized cutile: {opt_time:.3f} ms")
    print(f"CuPy (cuBLAS): {cupy_time:.3f} ms")
    print(f"Relative: {opt_time / cupy_time:.2f}x")

    print("\n--- All tests passed! ---")
