# SPDX-License-Identifier: Apache-2.0
"""
Causal Self-Attention - True Tile Programming Philosophy

This implementation follows the declarative Tile Programming approach:
- No explicit thread management
- High-level operations (matmul, reduce, broadcast)
- Compiler handles optimization
- Flash Attention style (online softmax)

Compare this with traditional CUDA where you'd manually:
- Manage shared memory for Q, K, V tiles
- Write reduction loops for softmax
- Insert __syncthreads() everywhere
- Handle thread indexing and bounds
"""

import math
import cupy as cp
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]


@ct.kernel
def causal_attention_tile_kernel(
    Q,              # Query: (batch, n_head, seq_len, head_dim)
    K,              # Key: (batch, n_head, seq_len, head_dim)
    V,              # Value: (batch, n_head, seq_len, head_dim)
    Out,            # Output: (batch, n_head, seq_len, head_dim)
    qk_scale: float,
    head_dim: ConstInt,
    n_head: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt
):
    """
    Declarative causal self-attention using Tile Philosophy.

    Uses Flash Attention approach (online softmax) to avoid materializing
    the full attention matrix.

    What we declare (WHAT):
    - Load Q, K, V tiles
    - Compute QK^T scores
    - Apply causal mask
    - Compute softmax in streaming fashion
    - Accumulate weighted values

    What compiler handles (HOW):
    - Thread to tile element mapping
    - Shared memory allocation
    - Synchronization between loads
    - Register allocation
    - Memory coalescing
    """
    # Get block position - NO thread IDs!
    bid_m = ct.bid(0)  # Query tile index
    bid_batch_head = ct.bid(1)  # Combined batch * head index

    # Decode batch and head indices
    batch_idx = bid_batch_head // n_head
    head_idx = bid_batch_head % n_head

    # ===========================
    # Step 1: Load Query tile
    # ===========================
    # Declarative: "Load this query tile"
    # Compiler handles: Thread assignments, coalescing
    q_tile = ct.load(
        Q,
        index=(batch_idx, head_idx, bid_m, 0),
        shape=(TILE_M, head_dim)
    )

    # ===========================
    # Step 2: Initialize accumulators for online softmax
    # ===========================
    # Online softmax maintains:
    # - m_i: running maximum
    # - l_i: running sum of exponentials
    # - acc: weighted sum of values
    m_i = ct.full((TILE_M, 1), -cp.inf, dtype=cp.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=cp.float32)
    acc = ct.full((TILE_M, head_dim), 0.0, dtype=cp.float32)

    # ===========================
    # Step 3: Causal attention loop
    # ===========================
    # Only attend to positions <= current position (causal mask)
    seq_len = K.shape[2]
    current_pos = (bid_m + 1) * TILE_M
    max_kv_tiles = ct.cdiv(min(current_pos, seq_len), TILE_N)

    # Create position offsets for masking
    # Declarative: "Create these index ranges"
    # Compiler handles: Constant generation
    offs_m = bid_m * TILE_M + ct.arange(TILE_M, dtype=np.int32)
    offs_m = offs_m[:, None]

    # Declarative loop over K, V tiles
    for kv_tile_idx in range(max_kv_tiles):
        # ===========================
        # Step 3a: Load K and V tiles
        # ===========================
        # Load K tile (will transpose for matmul)
        k_tile = ct.load(
            K,
            index=(batch_idx, head_idx, kv_tile_idx, 0),
            shape=(TILE_N, head_dim),
            order=(1, 0)  # Transpose: (head_dim, TILE_N)
        )

        v_tile = ct.load(
            V,
            index=(batch_idx, head_idx, kv_tile_idx, 0),
            shape=(TILE_N, head_dim)
        )

        # ===========================
        # Step 3b: Compute QK^T
        # ===========================
        # Declarative: "Multiply these matrices"
        # Compiler handles: MMA instruction selection, scheduling
        qk = ct.mma(q_tile, k_tile, ct.full((TILE_M, TILE_N), 0.0, dtype=cp.float32))

        # ===========================
        # Step 3c: Apply causal mask
        # ===========================
        # Declarative: "Mask future positions"
        # Compiler handles: Parallel comparison and selection
        offs_n = kv_tile_idx * TILE_N + ct.arange(TILE_N, dtype=np.int32)
        offs_n = offs_n[None, :]

        # Causal mask: query_pos >= key_pos
        mask = offs_m >= offs_n
        qk = ct.where(mask, qk * qk_scale, -cp.inf)

        # ===========================
        # Step 3d: Online softmax update
        # ===========================
        # Declarative softmax operations
        # Compiler handles: Reduction, broadcasting

        # New maximum
        m_ij = ct.max(qk, axis=-1, keepdims=True)
        m_ij_new = ct.maximum(m_i, m_ij)

        # Exponentials with corrected base
        qk_exp = ct.exp(qk - m_ij_new)

        # Update running sum
        exp_correction = ct.exp(m_i - m_ij_new)
        l_i = l_i * exp_correction + ct.sum(qk_exp, axis=-1, keepdims=True)

        # Update accumulator
        # Declarative: "Weight values by attention scores"
        acc = acc * exp_correction
        acc = ct.mma(qk_exp, v_tile, acc)

        # Update maximum
        m_i = m_ij_new

    # ===========================
    # Step 4: Final normalization
    # ===========================
    # Declarative: "Normalize by sum"
    # Compiler handles: Broadcasting division
    out_tile = acc / l_i

    # ===========================
    # Step 5: Store result
    # ===========================
    # Declarative: "Store this result"
    ct.store(Out, index=(batch_idx, head_idx, bid_m, 0), tile=out_tile)


def attention_tile(
    q: cp.ndarray,
    k: cp.ndarray,
    v: cp.ndarray,
    n_head: int
) -> cp.ndarray:
    """
    Apply causal self-attention using declarative Tile programming.

    This implements Flash Attention style computation:
    - Never materializes full attention matrix
    - Computes softmax in streaming/online fashion
    - Memory efficient O(seq_len) instead of O(seq_len^2)

    Args:
        q: Query tensor (batch, n_head, seq_len, head_dim)
        k: Key tensor (batch, n_head, seq_len, head_dim)
        v: Value tensor (batch, n_head, seq_len, head_dim)
        n_head: Number of attention heads

    Returns:
        Attention output (batch, n_head, seq_len, head_dim)
    """
    batch, n_head, seq_len, head_dim = q.shape

    # Ensure contiguous memory
    if not q.flags.c_contiguous:
        q = cp.ascontiguousarray(q)
    if not k.flags.c_contiguous:
        k = cp.ascontiguousarray(k)
    if not v.flags.c_contiguous:
        v = cp.ascontiguousarray(v)

    # Scale factor for attention
    qk_scale = 1.0 / math.sqrt(head_dim)

    # Output buffer
    out = cp.empty_like(q)

    # Tile sizes (powers of 2)
    TILE_M = 64
    TILE_N = 64

    # Grid dimensions
    grid_m = (seq_len + TILE_M - 1) // TILE_M
    grid_batch_head = batch * n_head
    grid = (grid_m, grid_batch_head)

    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        causal_attention_tile_kernel,
        (q, k, v, out, qk_scale, head_dim, n_head, TILE_M, TILE_N)
    )

    return out


def mha_tile(
    x: cp.ndarray,
    c_attn_weight: cp.ndarray,
    c_attn_bias: cp.ndarray,
    c_proj_weight: cp.ndarray,
    c_proj_bias: cp.ndarray,
    n_head: int
) -> cp.ndarray:
    """
    Full multi-head attention using Tile Philosophy kernels.

    This is the complete attention block as used in GPT, combining:
    1. QKV projection (linear)
    2. Multi-head attention
    3. Output projection (linear)

    Args:
        x: Input tensor (batch, seq_len, n_embd)
        c_attn_weight: QKV projection weight (3*n_embd, n_embd)
        c_attn_bias: QKV projection bias (3*n_embd,)
        c_proj_weight: Output projection weight (n_embd, n_embd)
        c_proj_bias: Output projection bias (n_embd,)
        n_head: Number of attention heads

    Returns:
        Output tensor (batch, seq_len, n_embd)
    """
    from .linear_tile import linear_tile

    batch, seq_len, n_embd = x.shape
    head_dim = n_embd // n_head

    # ===========================
    # Step 1: QKV Projection
    # ===========================
    # Declarative: "Project input to QKV"
    qkv = linear_tile(x, c_attn_weight, c_attn_bias)  # (B, T, 3*n_embd)

    # Split into Q, K, V
    q, k, v = cp.split(qkv, 3, axis=2)

    # ===========================
    # Step 2: Reshape for multi-head
    # ===========================
    # (batch, seq_len, n_embd) -> (batch, n_head, seq_len, head_dim)
    q = cp.transpose(
        cp.reshape(q, (batch, seq_len, n_head, head_dim)),
        (0, 2, 1, 3)
    )
    k = cp.transpose(
        cp.reshape(k, (batch, seq_len, n_head, head_dim)),
        (0, 2, 1, 3)
    )
    v = cp.transpose(
        cp.reshape(v, (batch, seq_len, n_head, head_dim)),
        (0, 2, 1, 3)
    )

    # ===========================
    # Step 3: Multi-head attention
    # ===========================
    # Declarative: "Apply causal attention"
    y = attention_tile(q, k, v, n_head)

    # ===========================
    # Step 4: Reshape back
    # ===========================
    # (batch, n_head, seq_len, head_dim) -> (batch, seq_len, n_embd)
    y = cp.transpose(y, (0, 2, 1, 3))
    if not y.flags.c_contiguous:
        y = cp.ascontiguousarray(y)
    y = cp.reshape(y, (batch, seq_len, n_embd))

    # ===========================
    # Step 5: Output projection
    # ===========================
    # Declarative: "Project back to embedding dimension"
    y = linear_tile(y, c_proj_weight, c_proj_bias)

    return y


# ============================================
# Comparison: Tile Philosophy vs Traditional
# ============================================

def attention_traditional_style():
    """
    Traditional CUDA attention would require:

    @cuda.jit
    def attention_kernel(Q, K, V, out, seq_len, head_dim):
        # Manual thread and block indexing
        tid = cuda.threadIdx.x
        bid_row = cuda.blockIdx.x
        bid_col = cuda.blockIdx.y

        # Shared memory for Q, K tiles
        __shared__ smem_q[BLOCK_SIZE][HEAD_DIM]
        __shared__ smem_k[BLOCK_SIZE][HEAD_DIM]
        __shared__ smem_scores[BLOCK_SIZE][BLOCK_SIZE]

        # Manual load Q tile
        for i in range(...):
            smem_q[tid][i] = Q[...]
        __syncthreads()

        # Loop over K tiles
        for k_tile in range(...):
            # Manual load K
            for i in range(...):
                smem_k[tid][i] = K[...]
            __syncthreads()

            # Manual QK^T computation
            score = 0
            for i in range(HEAD_DIM):
                score += smem_q[row][i] * smem_k[col][i]
            smem_scores[row][col] = score
            __syncthreads()

            # Manual softmax reduction
            # ... many lines of reduction code ...

            # Manual V accumulation
            # ... more manual work ...

    ❌ Hundreds of lines of low-level code
    ❌ Error-prone synchronization
    ❌ Hard to optimize
    ❌ Difficult to maintain
    """
    pass


# Our Tile approach:
# ✅ Declarative - specify WHAT not HOW
# ✅ Flash Attention - memory efficient online softmax
# ✅ Compiler optimizes - MMA selection, memory layout
# ✅ No synchronization - compiler handles dependencies
# ✅ Composable - easy to combine with other operations


if __name__ == "__main__":
    print("=== Testing Tile Philosophy Causal Attention ===\n")

    # Test dimensions
    batch, n_head, seq_len, head_dim = 2, 4, 128, 64
    n_embd = n_head * head_dim

    # Create test tensors
    q = cp.random.randn(batch, n_head, seq_len, head_dim, dtype=cp.float32)
    k = cp.random.randn(batch, n_head, seq_len, head_dim, dtype=cp.float32)
    v = cp.random.randn(batch, n_head, seq_len, head_dim, dtype=cp.float32)

    # Reference implementation (naive but correct)
    def reference_attention(q, k, v):
        scale = 1.0 / math.sqrt(head_dim)
        att = cp.matmul(q, cp.transpose(k, (0, 1, 3, 2))) * scale

        # Causal mask
        mask = cp.tril(cp.ones((seq_len, seq_len)))
        att = cp.where(mask == 0, float('-inf'), att)

        # Softmax
        att = cp.exp(att - cp.max(att, axis=-1, keepdims=True))
        att = att / cp.sum(att, axis=-1, keepdims=True)

        # Weighted sum
        return cp.matmul(att, v)

    # Tile implementation
    y_tile = attention_tile(q, k, v, n_head)
    y_ref = reference_attention(q, k, v)

    max_diff = cp.abs(y_tile - y_ref).max()
    print(f"Input shapes:")
    print(f"  Q: {q.shape}")
    print(f"  K: {k.shape}")
    print(f"  V: {v.shape}")
    print(f"Output shape: {y_tile.shape}")
    print(f"Max difference vs reference: {max_diff:.6f}")

    assert max_diff < 1e-2, f"Too large difference: {max_diff}"
    print("✅ Correctness test passed!")

    print("\n=== Key Tile Philosophy Principles ===")
    print("1. ✅ Declarative: High-level operations (mma, max, exp, sum)")
    print("2. ✅ No thread IDs: Compiler handles parallelization")
    print("3. ✅ No __syncthreads(): Compiler manages dependencies")
    print("4. ✅ Flash Attention: Memory-efficient online softmax")
    print("5. ✅ Composable: Easy to integrate into full transformer")

    print("\n=== Tile Philosophy: Think in Attention Operations, Not Threads! ===")
