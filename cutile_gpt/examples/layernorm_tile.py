# SPDX-License-Identifier: Apache-2.0
"""
LayerNorm - True Tile Programming Philosophy

This implementation follows the declarative Tile Programming approach:
- No explicit thread management
- No manual synchronization
- High-level operations (reduce, broadcast)
- Compiler handles optimization

Compare this with traditional CUDA where you'd manually:
- Manage shared memory
- Write reduction loops
- Insert __syncthreads()
- Handle thread indexing
"""

import cupy as cp
import cuda.tile as ct

ConstInt = ct.Constant[int]


@ct.kernel
def layernorm_tile_kernel(
    X,           # Input: (M, N)
    gamma,       # Weight: (N,)
    beta,        # Bias: (N,)
    Y,           # Output: (M, N)
    eps,         # Epsilon for stability
    N: ConstInt  # Normalized dimension
):
    """
    Declarative LayerNorm using Tile Philosophy.

    What we declare (WHAT):
    - Load a tile of data
    - Reduce to compute mean/variance
    - Normalize and transform

    What compiler handles (HOW):
    - Thread mapping
    - Shared memory
    - Synchronization
    - Register allocation
    """
    # Get block ID - but NO thread IDs!
    row_idx = ct.bid(0)

    # ===========================
    # Step 1: Load input tile
    # ===========================
    # Declarative: "Load this data"
    # Compiler decides: How to parallelize, which threads, what memory
    x_tile = ct.load(X, index=(row_idx, 0), shape=(1, N)).reshape((N,))

    # ===========================
    # Step 2: Compute statistics
    # ===========================
    # Declarative reduction - NO manual loop!
    # Compiler chooses optimal reduction algorithm
    total_sum = ct.sum(x_tile)
    mean = total_sum / N

    # Variance: E[X^2] - E[X]^2
    x_squared = x_tile * x_tile
    sum_squared = ct.sum(x_squared)
    variance = sum_squared / N - mean * mean

    # ===========================
    # Step 3: Normalize
    # ===========================
    # Declarative: WHAT we want
    # Compiler handles: HOW to broadcast, which threads compute what
    rstd = ct.rsqrt(variance + eps)  # 1 / sqrt(var + eps)

    # Broadcasting happens automatically - no manual expansion!
    x_centered = x_tile - mean
    x_normalized = x_centered * rstd

    # ===========================
    # Step 4: Affine transform
    # ===========================
    # Load parameters (compiler parallelizes)
    gamma_tile = ct.load(gamma, index=(0,), shape=(N,))
    beta_tile = ct.load(beta, index=(0,), shape=(N,))

    # Element-wise ops - automatic parallelization
    y_tile = x_normalized * gamma_tile + beta_tile

    # ===========================
    # Step 5: Store result
    # ===========================
    # Declarative: "Store this result"
    # Compiler decides: Coalescing pattern, memory ordering
    y_tile = y_tile.reshape((1, N))
    ct.store(Y, index=(row_idx, 0), tile=y_tile)


def layernorm_tile(
    x: cp.ndarray,
    weight: cp.ndarray,
    bias: cp.ndarray,
    eps: float = 1e-5
) -> cp.ndarray:
    """
    Apply LayerNorm using declarative Tile programming.

    This is the high-level interface that:
    1. Handles tensor shapes
    2. Launches the tile kernel
    3. Returns result

    The kernel itself is pure tile operations - no CUDA details!

    Args:
        x: Input tensor (..., normalized_dim)
        weight: Gamma parameter (normalized_dim,)
        bias: Beta parameter (normalized_dim,)
        eps: Numerical stability epsilon

    Returns:
        Normalized tensor with same shape as input
    """
    original_shape = x.shape
    n_embd = x.shape[-1]

    # Flatten to 2D: (batch * seq, n_embd)
    x_2d = x.reshape(-1, n_embd)
    M = x_2d.shape[0]

    # Ensure contiguous
    if not x_2d.flags.c_contiguous:
        x_2d = cp.ascontiguousarray(x_2d)

    # Output buffer
    y_2d = cp.empty_like(x_2d)

    # Launch: One block per row
    # Each block processes one row using tile operations
    grid = (M,)
    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        layernorm_tile_kernel,
        (x_2d, weight, bias, y_2d, eps, n_embd)
    )

    # Restore original shape
    return y_2d.reshape(original_shape)


# ============================================
# Comparison: Tile Philosophy vs Traditional
# ============================================

def layernorm_traditional_style(x, weight, bias, eps=1e-5):
    """
    How you might write it in traditional CUDA style:

    @cuda.jit
    def kernel(x, weight, bias, y, N):
        # Manual thread indexing
        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x

        # Manual shared memory allocation
        shared_sum = cuda.shared.array(256, dtype=float32)
        shared_sq = cuda.shared.array(256, dtype=float32)

        # Manual parallel load
        idx = bid * N + tid
        val = x[idx] if tid < N else 0

        # Manual reduction with explicit synchronization
        shared_sum[tid] = val
        shared_sq[tid] = val * val
        cuda.syncthreads()

        # Manual reduction tree
        s = 128
        while s > 0:
            if tid < s:
                shared_sum[tid] += shared_sum[tid + s]
                shared_sq[tid] += shared_sq[tid + s]
            cuda.syncthreads()
            s //= 2

        # ... more manual work ...

    ❌ Low-level, error-prone, hard to optimize
    """
    pass


def layernorm_torch_style(x, weight, bias, eps=1e-5):
    """
    How you write it in PyTorch:

    def forward(x, weight, bias):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + eps)
        return weight * x_norm + bias

    ✅ High-level but still imperative
    ✅ Framework handles GPU details
    ⚠️  Still specify HOW (mean, then var, then normalize)
    """
    pass


# Our Tile approach:
# ✅ Declarative - specify WHAT not HOW
# ✅ Compiler optimizes - reduction algorithm, memory layout
# ✅ No thread management - automatic parallelization
# ✅ No synchronization - compiler handles dependencies


if __name__ == "__main__":
    print("=== Testing Tile Philosophy LayerNorm ===\n")

    # Test dimensions
    batch, seq, n_embd = 4, 128, 768

    # Create test data
    x = cp.random.randn(batch, seq, n_embd, dtype=cp.float32)
    weight = cp.ones(n_embd, dtype=cp.float32)
    bias = cp.zeros(n_embd, dtype=cp.float32)
    eps = 1e-5

    # Tile Philosophy implementation
    y_tile = layernorm_tile(x, weight, bias, eps)

    # Reference (CuPy)
    mean = cp.mean(x, axis=-1, keepdims=True)
    var = cp.var(x, axis=-1, keepdims=True)
    y_ref = weight * (x - mean) / cp.sqrt(var + eps) + bias

    # Check correctness
    max_diff = cp.abs(y_tile - y_ref).max()
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_tile.shape}")
    print(f"Max difference vs reference: {max_diff:.6f}")

    # Numerical test
    assert max_diff < 1e-4, f"Too large difference: {max_diff}"
    print("✅ Correctness test passed!")

    print("\n=== Key Tile Philosophy Principles ===")
    print("1. ✅ Declarative: Only WHAT we want (reduce, normalize)")
    print("2. ✅ No thread IDs: Compiler handles parallelization")
    print("3. ✅ No __syncthreads(): Compiler manages dependencies")
    print("4. ✅ No shared memory: Compiler chooses memory strategy")
    print("5. ✅ High-level ops: reduce, broadcast, element-wise")

    print("\n=== Tile Philosophy: Think in Data, Not Threads! ===")
