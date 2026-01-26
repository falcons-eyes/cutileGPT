# SPDX-License-Identifier: Apache-2.0
"""
Linear layer kernel V2 with Tile IR advanced features.

New optimizations:
- Tensor views with shape/stride information
- Structured loop constructs with loop-carried variables
- Extended optimization hints
- Explicit alignment assumptions

This is a proof-of-concept implementation demonstrating Tile IR
advanced features from the official documentation.
"""

import math
import cupy as cp
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Swizzle pattern for better L2 cache locality
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


@ct.kernel(
    num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1),
    occupancy=4,
    # Extended optimization hints (Tile IR feature)
    optimization_hints={
        'max_register_usage': 128,      # Control register pressure
        'prefer_l1_cache': True,        # L1 vs shared memory trade-off
        'vectorization_factor': 4,      # SIMD width hint
        'unroll_factor': 2,             # Loop unrolling hint
    }
)
def matmul_kernel_v2(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    """
    Matrix multiplication V2 with Tile IR advanced features.

    Key improvements over V1:
    1. Tensor views for better compiler optimization
    2. Structured loop with explicit loop-carried variables
    3. Extended optimization hints
    4. Alignment assumptions

    Args:
        A: Input matrix (M, K)
        B: Weight matrix (K, N)
        C: Output matrix (M, N)
        tm, tn, tk: Tile sizes
    """
    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]

    # Create tensor views with shape/stride information
    # This enables compiler to optimize memory access patterns
    A_view = ct.make_tensor_view(A, shape=(M, K), strides=(K, 1))
    B_view = ct.make_tensor_view(B, shape=(K, N), strides=(N, 1))
    C_view = ct.make_tensor_view(C, shape=(M, N), strides=(N, 1))

    # Explicit alignment assumptions for vectorization
    ct.assume(A_view.is_aligned(16))  # 128-bit alignment
    ct.assume(B_view.is_aligned(16))
    ct.assume(C_view.is_aligned(16))

    # 2D swizzle for L2 cache locality
    bid_m, bid_n = swizzle_2d(M, N, tm, tn)

    num_tiles_k = ct.cdiv(K, tk)

    # Structured loop with explicit loop-carried variables
    # This makes the accumulation pattern explicit for the compiler
    loop_state = ct.create_loop_state(
        acc=ct.full((tm, tn), 0, dtype=ct.float32)
    )

    # Use TF32 for tensor cores with float32 input
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for k in ct.loop_range(num_tiles_k, carried_vars=loop_state):
        # Load with tensor views and extended hints
        a = ct.load_view(
            A_view,
            tile_idx=(bid_m, k),
            tile_shape=(tm, tk),
            latency=4,
            allow_tma=True,
            prefetch_distance=2,          # Prefetch ahead
            cache_policy='streaming'       # Streaming vs persistent
        ).astype(dtype)

        b = ct.load_view(
            B_view,
            tile_idx=(k, bid_n),
            tile_shape=(tk, tn),
            latency=4,
            allow_tma=True,
            prefetch_distance=2,
            cache_policy='streaming'
        ).astype(dtype)

        # Matrix multiply-accumulate
        new_acc = ct.mma(a, b, loop_state.acc)

        # Explicit continue with updated state
        loop_state = ct.continue_loop(acc=new_acc)

    # Extract final accumulator value
    final_acc = loop_state.extract().acc

    # Store with tensor view
    ct.store_view(C_view, tile_idx=(bid_m, bid_n), tile=final_acc.astype(C.dtype))


@ct.kernel(
    num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1),
    occupancy=4,
    optimization_hints={
        'max_register_usage': 128,
        'prefer_l1_cache': True,
        'vectorization_factor': 4,
        'unroll_factor': 2,
    }
)
def matmul_bias_kernel_v2(A, B, bias, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    """
    Fused matrix multiplication with bias V2.

    Uses Tile IR advanced features for optimal performance.
    """
    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]

    # Tensor views
    A_view = ct.make_tensor_view(A, shape=(M, K), strides=(K, 1))
    B_view = ct.make_tensor_view(B, shape=(K, N), strides=(N, 1))
    bias_view = ct.make_tensor_view(bias, shape=(N,), strides=(1,))
    C_view = ct.make_tensor_view(C, shape=(M, N), strides=(N, 1))

    # Alignment assumptions
    ct.assume(A_view.is_aligned(16))
    ct.assume(B_view.is_aligned(16))
    ct.assume(bias_view.is_aligned(16))
    ct.assume(C_view.is_aligned(16))

    bid_m, bid_n = swizzle_2d(M, N, tm, tn)
    num_tiles_k = ct.cdiv(K, tk)

    # Structured loop
    loop_state = ct.create_loop_state(
        acc=ct.full((tm, tn), 0, dtype=ct.float32)
    )

    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for k in ct.loop_range(num_tiles_k, carried_vars=loop_state):
        a = ct.load_view(
            A_view,
            tile_idx=(bid_m, k),
            tile_shape=(tm, tk),
            latency=4,
            allow_tma=True,
            prefetch_distance=2,
            cache_policy='streaming'
        ).astype(dtype)

        b = ct.load_view(
            B_view,
            tile_idx=(k, bid_n),
            tile_shape=(tk, tn),
            latency=4,
            allow_tma=True,
            prefetch_distance=2,
            cache_policy='streaming'
        ).astype(dtype)

        new_acc = ct.mma(a, b, loop_state.acc)
        loop_state = ct.continue_loop(acc=new_acc)

    acc = loop_state.extract().acc

    # Fused bias addition
    b_tile = ct.load_view(
        bias_view,
        tile_idx=(bid_n,),
        tile_shape=(tn,),
        latency=2,
        allow_tma=True
    )
    acc = acc + b_tile

    ct.store_view(C_view, tile_idx=(bid_m, bid_n), tile=acc.astype(C.dtype))


def cutile_linear_v2(x: cp.ndarray, weight: cp.ndarray, weight_t: cp.ndarray = None) -> cp.ndarray:
    """
    Linear transformation V2 with Tile IR advanced features.

    This version uses tensor views, structured loops, and extended hints
    for better compiler optimization.

    Args:
        x: Input tensor (..., in_features)
        weight: Weight matrix (out_features, in_features)
        weight_t: Optional pre-transposed weight (in_features, out_features)

    Returns:
        Output tensor (..., out_features)
    """
    if not isinstance(x, cp.ndarray) or not isinstance(weight, cp.ndarray):
        raise ValueError("Tensors must be CuPy arrays on CUDA device")

    original_shape = x.shape[:-1]
    in_features = x.shape[-1]
    out_features = weight.shape[0]

    # Reshape x to 2D
    x_2d = cp.reshape(x, (-1, in_features))
    if not x_2d.flags.c_contiguous:
        x_2d = cp.ascontiguousarray(x_2d)
    M = x_2d.shape[0]
    N = out_features

    # Use pre-computed transpose if available
    if weight_t is None:
        weight_t = cp.transpose(weight)
        if not weight_t.flags.c_contiguous:
            weight_t = cp.ascontiguousarray(weight_t)

    # Output
    output = cp.empty((M, N), dtype=x.dtype)

    # Tile sizes
    if x.dtype in (cp.float16, cp.dtype('float16')):
        tm, tn, tk = 128, 128, 64
    else:
        tm, tn, tk = 32, 32, 32

    grid_m = math.ceil(M / tm)
    grid_n = math.ceil(N / tn)
    grid = (grid_m * grid_n, 1, 1)

    ct.launch(cp.cuda.get_current_stream(), grid, matmul_kernel_v2,
              (x_2d, weight_t, output, tm, tn, tk))

    return cp.reshape(output, (*original_shape, out_features))


def cutile_linear_bias_v2(
    x: cp.ndarray,
    weight: cp.ndarray,
    bias: cp.ndarray,
    weight_t: cp.ndarray = None
) -> cp.ndarray:
    """
    Fused linear transformation with bias V2.

    Uses Tile IR advanced features for optimal performance.

    Args:
        x: Input tensor (..., in_features)
        weight: Weight matrix (out_features, in_features)
        bias: Bias vector (out_features,)
        weight_t: Optional pre-transposed weight

    Returns:
        Output tensor (..., out_features)
    """
    if not isinstance(x, cp.ndarray) or not isinstance(weight, cp.ndarray):
        raise ValueError("Tensors must be CuPy arrays on CUDA device")

    original_shape = x.shape[:-1]
    in_features = x.shape[-1]
    out_features = weight.shape[0]

    # Reshape x to 2D
    x_2d = cp.reshape(x, (-1, in_features))
    if not x_2d.flags.c_contiguous:
        x_2d = cp.ascontiguousarray(x_2d)
    M = x_2d.shape[0]
    N = out_features

    # Use pre-computed transpose
    if weight_t is None:
        weight_t = cp.transpose(weight)
        if not weight_t.flags.c_contiguous:
            weight_t = cp.ascontiguousarray(weight_t)

    # Output
    output = cp.empty((M, N), dtype=x.dtype)

    # Tile sizes
    if x.dtype in (cp.float16, cp.dtype('float16')):
        tm, tn, tk = 128, 128, 64
    else:
        tm, tn, tk = 32, 32, 32

    grid_m = math.ceil(M / tm)
    grid_n = math.ceil(N / tn)
    grid = (grid_m * grid_n, 1, 1)

    ct.launch(cp.cuda.get_current_stream(), grid, matmul_bias_kernel_v2,
              (x_2d, weight_t, bias, output, tm, tn, tk))

    return cp.reshape(output, (*original_shape, out_features))


# Reference implementation using CuPy
def cupy_linear(x: cp.ndarray, weight: cp.ndarray,
                bias: cp.ndarray = None) -> cp.ndarray:
    """CuPy reference linear: y = x @ weight.T + bias"""
    y = cp.matmul(x, weight.T)
    if bias is not None:
        y = y + bias
    return y


if __name__ == "__main__":
    print("--- Testing cutile Linear V2 (Tile IR Advanced Features) ---\n")

    batch, seq, in_feat, out_feat = 8, 128, 128, 512

    x = cp.random.randn(batch, seq, in_feat, dtype=cp.float32)
    weight = cp.random.randn(out_feat, in_feat, dtype=cp.float32)
    bias = cp.random.randn(out_feat, dtype=cp.float32)

    # Precompute transpose
    weight_t = cp.transpose(weight)
    if not weight_t.flags.c_contiguous:
        weight_t = cp.ascontiguousarray(weight_t)

    print("Testing correctness...")

    # Test without bias
    y_cutile_v2 = cutile_linear_v2(x, weight, weight_t)
    y_cupy = cupy_linear(x, weight)

    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Output shape: {y_cutile_v2.shape}")
    print(f"Without bias - Max diff: {cp.abs(y_cutile_v2 - y_cupy).max():.6f}")
    cp.testing.assert_allclose(y_cutile_v2, y_cupy, atol=1e-4, rtol=1e-4)
    print("✓ Linear (no bias) test passed!")

    # Test with bias
    y_cutile_v2_bias = cutile_linear_bias_v2(x, weight, bias, weight_t)
    y_cupy_bias = cupy_linear(x, weight, bias)

    print(f"With bias - Max diff: {cp.abs(y_cutile_v2_bias - y_cupy_bias).max():.6f}")
    cp.testing.assert_allclose(y_cutile_v2_bias, y_cupy_bias, atol=1e-4, rtol=1e-4)
    print("✓ Linear (with bias) test passed!")

    print("\n--- Benchmarking V1 vs V2 ---")
    import time
    from .linear import cutile_linear_bias

    # Warmup
    for _ in range(10):
        _ = cutile_linear_bias(x, weight, bias, weight_t)
        _ = cutile_linear_bias_v2(x, weight, bias, weight_t)
    cp.cuda.Stream.null.synchronize()

    # Benchmark V1
    start = time.time()
    for _ in range(100):
        _ = cutile_linear_bias(x, weight, bias, weight_t)
    cp.cuda.Stream.null.synchronize()
    v1_time = (time.time() - start) / 100 * 1000

    # Benchmark V2
    start = time.time()
    for _ in range(100):
        _ = cutile_linear_bias_v2(x, weight, bias, weight_t)
    cp.cuda.Stream.null.synchronize()
    v2_time = (time.time() - start) / 100 * 1000

    # Benchmark CuPy
    start = time.time()
    for _ in range(100):
        _ = cupy_linear(x, weight, bias)
    cp.cuda.Stream.null.synchronize()
    cupy_time = (time.time() - start) / 100 * 1000

    print(f"\nResults (batch={batch}, seq={seq}, in={in_feat}, out={out_feat}):")
    print(f"  cutile V1:  {v1_time:.3f} ms")
    print(f"  cutile V2:  {v2_time:.3f} ms  (speedup: {v1_time/v2_time:.2f}x)")
    print(f"  CuPy:       {cupy_time:.3f} ms")
    print(f"\nV2 vs CuPy: {v2_time/cupy_time:.2f}x")

    if v2_time < v1_time:
        print(f"\n✅ V2 is {(v1_time - v2_time) / v1_time * 100:.1f}% faster than V1!")
    else:
        print(f"\n⚠️  V2 is {(v2_time - v1_time) / v1_time * 100:.1f}% slower (may need tuning)")

    print("\n--- All tests passed! ---")
