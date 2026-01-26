# SPDX-License-Identifier: Apache-2.0
"""
GELU - True Tile Programming Philosophy

This implementation follows the declarative Tile Programming approach:
- No explicit thread management
- No manual synchronization
- Pure element-wise tile operations
- Compiler handles optimization

Compare with traditional approaches:
- CUDA: Manual thread indexing, shared memory, __syncthreads()
- PyTorch: Imperative but framework handles GPU
- Tile: Declarative WHAT, compiler handles HOW
"""

import math
import cupy as cp
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Constants for GELU approximation (GPT-2 style)
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)  # ~0.7978845608
GELU_COEF = 0.044715


@ct.kernel
def gelu_tile_kernel(X, Y, N: ConstInt):
    """
    Declarative GELU activation using Tile Philosophy.

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    What we declare (WHAT):
    - Load input tile
    - Apply element-wise mathematical transformations
    - Store result

    What compiler handles (HOW):
    - Thread mapping to elements
    - Memory coalescing
    - Register allocation
    - Instruction scheduling
    """
    # Get block ID - but NO thread IDs!
    idx = ct.bid(0)

    # ===========================
    # Step 1: Load input tile
    # ===========================
    # Declarative: "Load this data"
    # Compiler decides: Which threads load what, memory access pattern
    x = ct.load(X, index=(idx,), shape=(N,))

    # ===========================
    # Step 2: GELU computation
    # ===========================
    # Declarative element-wise operations
    # Compiler handles: Parallelization across elements

    # x^3
    x_cubed = x * x * x

    # sqrt(2/pi) * (x + 0.044715 * x^3)
    inner = SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed)

    # tanh(inner)
    tanh_inner = ct.tanh(inner)

    # 0.5 * x * (1 + tanh(inner))
    y = 0.5 * x * (1.0 + tanh_inner)

    # ===========================
    # Step 3: Store result
    # ===========================
    # Declarative: "Store this result"
    # Compiler decides: Write pattern, coalescing strategy
    ct.store(Y, index=(idx,), tile=y)


def gelu_tile(x: cp.ndarray) -> cp.ndarray:
    """
    Apply GELU activation using declarative Tile programming.

    This is the high-level interface that:
    1. Handles tensor shapes
    2. Launches the tile kernel
    3. Returns result

    The kernel itself is pure tile operations - no CUDA details!

    Args:
        x: Input tensor of any shape

    Returns:
        Tensor with GELU applied element-wise (same shape as input)
    """
    original_shape = x.shape

    # Flatten to 1D for processing
    x_flat = x.reshape(-1)
    total_elements = x_flat.size

    # Ensure contiguous
    if not x_flat.flags.c_contiguous:
        x_flat = cp.ascontiguousarray(x_flat)

    # Output buffer
    y_flat = cp.empty_like(x_flat)

    # Choose tile size (power of 2 for efficiency)
    TILE_SIZE = 1024

    # Launch: One block per tile
    # Each block processes TILE_SIZE elements using tile operations
    num_blocks = (total_elements + TILE_SIZE - 1) // TILE_SIZE
    grid = (num_blocks,)

    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        gelu_tile_kernel,
        (x_flat, y_flat, TILE_SIZE)
    )

    # Restore original shape
    return y_flat.reshape(original_shape)


# ============================================
# Comparison: Tile Philosophy vs Traditional
# ============================================

def gelu_traditional_style(x):
    """
    How you might write it in traditional CUDA style:

    @cuda.jit
    def kernel(x, y, N):
        # Manual thread indexing
        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        idx = bid * cuda.blockDim.x + tid

        if idx < N:
            # Manual computation
            val = x[idx]
            x3 = val * val * val
            inner = SQRT_2_OVER_PI * (val + GELU_COEF * x3)
            y[idx] = 0.5 * val * (1.0 + tanh(inner))

    ❌ Low-level, manual thread management
    ❌ Easy to make indexing errors
    ❌ Hard to optimize across different GPUs
    """
    pass


def gelu_torch_style(x):
    """
    How you write it in PyTorch:

    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))

    ✅ High-level and readable
    ✅ Framework handles GPU details
    ⚠️  Still imperative (specify HOW to compute)
    ⚠️  Framework overhead
    """
    pass


# Our Tile approach:
# ✅ Declarative - specify WHAT not HOW
# ✅ Compiler optimizes - element-wise parallelization
# ✅ No thread management - automatic distribution
# ✅ No synchronization - no shared state


# Reference CuPy implementation for testing
def cupy_gelu(x: cp.ndarray) -> cp.ndarray:
    """CuPy reference GELU (approximate version matching GPT-2)"""
    return 0.5 * x * (1.0 + cp.tanh(SQRT_2_OVER_PI * (x + GELU_COEF * cp.power(x, 3.0))))


if __name__ == "__main__":
    print("=== Testing Tile Philosophy GELU ===\n")

    # Test 1: 1D tensor
    print("Test 1: 1D tensor")
    x = cp.random.randn(2048, dtype=cp.float32)
    y_tile = gelu_tile(x)
    y_ref = cupy_gelu(x)

    max_diff = cp.abs(y_tile - y_ref).max()
    print(f"  Shape: {x.shape}")
    print(f"  Max difference vs reference: {max_diff:.6f}")
    assert max_diff < 1e-5, f"Too large difference: {max_diff}"
    print("  ✅ Test passed!")

    # Test 2: 2D tensor (matrix)
    print("\nTest 2: 2D tensor")
    x = cp.random.randn(128, 256, dtype=cp.float32)
    y_tile = gelu_tile(x)
    y_ref = cupy_gelu(x)

    max_diff = cp.abs(y_tile - y_ref).max()
    print(f"  Shape: {x.shape}")
    print(f"  Max difference vs reference: {max_diff:.6f}")
    assert max_diff < 1e-5, f"Too large difference: {max_diff}"
    print("  ✅ Test passed!")

    # Test 3: 3D tensor (typical transformer: batch, seq, hidden)
    print("\nTest 3: 3D tensor (transformer shape)")
    batch, seq, hidden = 4, 128, 768
    x = cp.random.randn(batch, seq, hidden, dtype=cp.float32)
    y_tile = gelu_tile(x)
    y_ref = cupy_gelu(x)

    max_diff = cp.abs(y_tile - y_ref).max()
    print(f"  Shape: {x.shape}")
    print(f"  Max difference vs reference: {max_diff:.6f}")
    assert max_diff < 1e-5, f"Too large difference: {max_diff}"
    print("  ✅ Test passed!")

    print("\n=== Key Tile Philosophy Principles ===")
    print("1. ✅ Declarative: Only WHAT we want (element-wise ops)")
    print("2. ✅ No thread IDs: Compiler handles parallelization")
    print("3. ✅ No synchronization: Pure element-wise, no shared state")
    print("4. ✅ High-level math: tanh, multiply - compiler optimizes")
    print("5. ✅ Portable: Same code works on any GPU")

    print("\n=== Tile Philosophy: Think in Operations, Not Threads! ===")
