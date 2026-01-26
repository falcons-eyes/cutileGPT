# SPDX-License-Identifier: Apache-2.0
"""
Linear (MatMul) - True Tile Programming Philosophy

This implementation follows the declarative Tile Programming approach:
- No explicit thread management
- No manual tile accumulation loops
- High-level matrix multiplication
- Compiler handles optimization

Compare this with traditional CUDA where you'd manually:
- Manage shared memory tiles
- Write nested loops for tile loading
- Insert __syncthreads() for tile boundaries
- Handle edge cases and padding
"""

import cupy as cp
import cuda.tile as ct

ConstInt = ct.Constant[int]


@ct.kernel
def matmul_tile_kernel(
    A,           # Input: (M, K)
    B,           # Weight: (K, N)
    C,           # Output: (M, N)
    M: ConstInt,
    N: ConstInt,
    K: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt
):
    """
    Declarative matrix multiplication using Tile Philosophy.

    Computes C = A @ B

    What we declare (WHAT):
    - Load tiles from A and B
    - Multiply and accumulate
    - Store result tile

    What compiler handles (HOW):
    - Thread to tile element mapping
    - Shared memory allocation
    - Synchronization between tile loads
    - Register allocation
    - Memory coalescing
    """
    # Get block position - NO thread IDs!
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # ===========================
    # Step 1: Initialize accumulator
    # ===========================
    # Declarative: Create a tile filled with zeros
    # Compiler handles: Register allocation
    acc = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)

    # ===========================
    # Step 2: Tile-based matrix multiplication
    # ===========================
    # Declarative loop over K dimension
    # Compiler handles: Tile loading, synchronization
    num_k_tiles = ct.cdiv(K, TILE_K)

    for k_tile in range(num_k_tiles):
        # Load tile from A: (TILE_M, TILE_K)
        # Declarative: "Load this tile"
        # Compiler: Handles which threads load what, coalescing
        a_tile = ct.load(
            A,
            index=(bid_m, k_tile),
            shape=(TILE_M, TILE_K)
        )

        # Load tile from B: (TILE_K, TILE_N)
        # Declarative: "Load this tile"
        b_tile = ct.load(
            B,
            index=(k_tile, bid_n),
            shape=(TILE_K, TILE_N)
        )

        # Matrix multiply-accumulate
        # Declarative: "Multiply these tiles and accumulate"
        # Compiler handles: Optimal MMA instruction selection,
        #                   register usage, instruction scheduling
        acc = ct.mma(a_tile, b_tile, acc)

        # NO explicit __syncthreads() needed!
        # Compiler manages dependencies between iterations

    # ===========================
    # Step 3: Store result
    # ===========================
    # Declarative: "Store this tile"
    # Compiler decides: Write pattern, coalescing
    ct.store(C, index=(bid_m, bid_n), tile=acc)


@ct.kernel
def matmul_bias_tile_kernel(
    A,           # Input: (M, K)
    B,           # Weight: (K, N)
    bias,        # Bias: (N,)
    C,           # Output: (M, N)
    M: ConstInt,
    N: ConstInt,
    K: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt
):
    """
    Declarative matrix multiplication with bias addition.

    Computes C = A @ B + bias

    Demonstrates automatic broadcasting in Tile Programming.
    """
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # Initialize accumulator
    acc = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)

    # Matrix multiplication
    num_k_tiles = ct.cdiv(K, TILE_K)
    for k_tile in range(num_k_tiles):
        a_tile = ct.load(A, index=(bid_m, k_tile), shape=(TILE_M, TILE_K))
        b_tile = ct.load(B, index=(k_tile, bid_n), shape=(TILE_K, TILE_N))
        acc = ct.mma(a_tile, b_tile, acc)

    # Load bias tile
    # Declarative: "Load this bias"
    bias_tile = ct.load(bias, index=(bid_n,), shape=(TILE_N,))

    # Add bias with automatic broadcasting
    # Declarative: "Add bias to each row"
    # Compiler handles: Broadcasting (TILE_N,) to (TILE_M, TILE_N)
    result = acc + bias_tile

    # Store result
    ct.store(C, index=(bid_m, bid_n), tile=result)


def linear_tile(
    x: cp.ndarray,
    weight: cp.ndarray,
    bias: cp.ndarray = None
) -> cp.ndarray:
    """
    Apply linear transformation using declarative Tile programming.

    y = x @ weight.T + bias

    This is the high-level interface that:
    1. Handles tensor shapes
    2. Transposes weight (PyTorch convention)
    3. Launches the tile kernel
    4. Returns result

    The kernel itself is pure tile operations - no CUDA details!

    Args:
        x: Input tensor (..., in_features)
        weight: Weight matrix (out_features, in_features)
        bias: Optional bias vector (out_features,)

    Returns:
        Output tensor (..., out_features)
    """
    # Handle arbitrary input shapes
    original_shape = x.shape
    in_features = original_shape[-1]
    out_features = weight.shape[0]

    # Flatten to 2D: (...) -> (M, K)
    x_2d = x.reshape(-1, in_features)
    M = x_2d.shape[0]
    K = in_features
    N = out_features

    # Weight is (out_features, in_features), need (K, N)
    weight_t = weight.T

    # Ensure contiguous
    if not x_2d.flags.c_contiguous:
        x_2d = cp.ascontiguousarray(x_2d)
    if not weight_t.flags.c_contiguous:
        weight_t = cp.ascontiguousarray(weight_t)

    # Output buffer
    y_2d = cp.empty((M, N), dtype=x.dtype)

    # Tile sizes (powers of 2 for efficiency)
    TILE_M = 64
    TILE_N = 64
    TILE_K = 32

    # Grid dimensions
    grid_m = (M + TILE_M - 1) // TILE_M
    grid_n = (N + TILE_N - 1) // TILE_N
    grid = (grid_m, grid_n)

    # Launch appropriate kernel
    if bias is not None:
        # Ensure bias is contiguous
        if not bias.flags.c_contiguous:
            bias = cp.ascontiguousarray(bias)

        ct.launch(
            cp.cuda.get_current_stream(),
            grid,
            matmul_bias_tile_kernel,
            (x_2d, weight_t, bias, y_2d, M, N, K, TILE_M, TILE_N, TILE_K)
        )
    else:
        ct.launch(
            cp.cuda.get_current_stream(),
            grid,
            matmul_tile_kernel,
            (x_2d, weight_t, y_2d, M, N, K, TILE_M, TILE_N, TILE_K)
        )

    # Restore original shape
    output_shape = original_shape[:-1] + (out_features,)
    return y_2d.reshape(output_shape)


# ============================================
# Comparison: Tile Philosophy vs Traditional
# ============================================

def matmul_traditional_style(A, B, C):
    """
    How you might write it in traditional CUDA style:

    @cuda.jit
    def kernel(A, B, C, M, N, K):
        # Manual thread indexing
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y

        # Manual shared memory allocation
        TILE = 16
        As = cuda.shared.array((TILE, TILE), dtype=float32)
        Bs = cuda.shared.array((TILE, TILE), dtype=float32)

        # Compute global position
        row = by * TILE + ty
        col = bx * TILE + tx

        # Accumulator
        acc = 0.0

        # Manual tile loop
        for k_tile in range(0, K, TILE):
            # Manual cooperative load to shared memory
            if row < M and k_tile + tx < K:
                As[ty, tx] = A[row, k_tile + tx]
            else:
                As[ty, tx] = 0.0

            if k_tile + ty < K and col < N:
                Bs[ty, tx] = B[k_tile + ty, col]
            else:
                Bs[ty, tx] = 0.0

            # Explicit synchronization
            cuda.syncthreads()

            # Manual accumulation
            for k in range(TILE):
                acc += As[ty, k] * Bs[k, tx]

            # Explicit synchronization
            cuda.syncthreads()

        # Write result
        if row < M and col < N:
            C[row, col] = acc

    ❌ Low-level, error-prone
    ❌ Manual synchronization
    ❌ Manual memory management
    ❌ Hard to tune for different GPUs
    """
    pass


def matmul_torch_style(x, weight, bias):
    """
    How you write it in PyTorch:

    def linear(x, weight, bias):
        y = torch.matmul(x, weight.T)
        if bias is not None:
            y = y + bias
        return y

    ✅ High-level and simple
    ✅ Framework handles GPU details
    ⚠️  Still imperative (matmul then add)
    ⚠️  Framework overhead
    ⚠️  Limited control over optimization
    """
    pass


# Our Tile approach:
# ✅ Declarative - specify WHAT not HOW
# ✅ Compiler optimizes - MMA instruction selection, tiling strategy
# ✅ No thread management - automatic parallelization
# ✅ No synchronization - compiler handles dependencies
# ✅ Composable - bias addition via broadcasting


if __name__ == "__main__":
    print("=== Testing Tile Philosophy Linear ===\n")

    # Test 1: Simple matrix multiplication
    print("Test 1: Matrix multiplication")
    M, K, N = 128, 64, 96
    A = cp.random.randn(M, K, dtype=cp.float32)
    B = cp.random.randn(N, K, dtype=cp.float32)  # (out_features, in_features)

    C_tile = linear_tile(A, B)
    C_ref = A @ B.T

    max_diff = cp.abs(C_tile - C_ref).max()
    print(f"  Input shape: {A.shape}")
    print(f"  Weight shape: {B.shape}")
    print(f"  Output shape: {C_tile.shape}")
    print(f"  Max difference vs reference: {max_diff:.6f}")
    assert max_diff < 1e-3, f"Too large difference: {max_diff}"
    print("  ✅ Test passed!")

    # Test 2: With bias
    print("\nTest 2: Matrix multiplication with bias")
    bias = cp.random.randn(N, dtype=cp.float32)

    C_tile = linear_tile(A, B, bias)
    C_ref = A @ B.T + bias

    max_diff = cp.abs(C_tile - C_ref).max()
    print(f"  Output shape: {C_tile.shape}")
    print(f"  Max difference vs reference: {max_diff:.6f}")
    assert max_diff < 1e-3, f"Too large difference: {max_diff}"
    print("  ✅ Test passed!")

    # Test 3: 3D tensor (batch processing)
    print("\nTest 3: 3D tensor (transformer shape)")
    batch, seq, in_features = 4, 128, 768
    out_features = 3072

    x = cp.random.randn(batch, seq, in_features, dtype=cp.float32)
    weight = cp.random.randn(out_features, in_features, dtype=cp.float32)
    bias = cp.random.randn(out_features, dtype=cp.float32)

    y_tile = linear_tile(x, weight, bias)
    y_ref = cp.reshape(
        cp.reshape(x, (-1, in_features)) @ weight.T + bias,
        (batch, seq, out_features)
    )

    max_diff = cp.abs(y_tile - y_ref).max()
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y_tile.shape}")
    print(f"  Max difference vs reference: {max_diff:.6f}")
    assert max_diff < 1e-3, f"Too large difference: {max_diff}"
    print("  ✅ Test passed!")

    print("\n=== Key Tile Philosophy Principles ===")
    print("1. ✅ Declarative: High-level mma operation")
    print("2. ✅ No thread IDs: Compiler handles parallelization")
    print("3. ✅ No __syncthreads(): Compiler manages dependencies")
    print("4. ✅ No shared memory: Compiler chooses memory strategy")
    print("5. ✅ Composable: Easy to add bias via broadcasting")

    print("\n=== Tile Philosophy: Think in Matrix Operations, Not Threads! ===")
