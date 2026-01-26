# SPDX-License-Identifier: Apache-2.0
"""
Improved Causal Self-Attention with explicit loop structure.

Step-by-step improvements:
1. More explicit accumulator management
2. Loop unrolling hints via structure
3. Better prefetch patterns
"""

import math
import cupy as cp
import cuda.tile as ct
from cuda.tile import RoundingMode as RMd
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

INV_LOG_2 = 1.0 / math.log(2)


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def causal_attention_kernel_v2(
    Q, K, V, Out,
    qk_scale: float,
    TILE_D: ConstInt,
    N_HEAD: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt
):
    """
    Improved causal attention with more explicit structure.

    Key improvements:
    1. Explicit accumulator state management
    2. Separated load and compute phases for better pipelining
    3. More structured loop body
    """
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)

    batch_idx = bid_y // N_HEAD
    head_idx = bid_y % N_HEAD

    qk_scale_log2 = qk_scale * INV_LOG_2

    # Query position offsets
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=np.int32)
    offs_m = offs_m[:, None]

    # Key/Value position offsets
    offs_n_tile = ct.arange(TILE_N, dtype=np.int32)
    offs_n_tile = offs_n_tile[None, :]

    # Initialize accumulators explicitly
    # These are the loop-carried variables in Flash Attention
    max_val = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)  # m_i in paper
    sum_exp = ct.full((TILE_M, 1), 0.0, dtype=np.float32)      # l_i in paper
    output = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)  # O_i in paper

    # Load query once (reused across all K/V tiles)
    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0),
                shape=(1, 1, TILE_M, TILE_D), latency=4, allow_tma=True).reshape((TILE_M, TILE_D))

    # Causal masking boundary
    seq_len = K.shape[2]
    m_end = (bid_x + 1) * TILE_M
    num_kv_tiles = ct.cdiv(min(m_end, seq_len), TILE_N)

    # ===== Online Softmax Loop =====
    # This is the critical loop for Flash Attention performance
    for kv_tile_idx in range(0, num_kv_tiles):
        # ---- Phase 1: Load K and compute QK^T ----
        k = ct.load(K, index=(batch_idx, head_idx, 0, kv_tile_idx),
                    shape=(1, 1, TILE_D, TILE_N),
                    order=(0, 1, 3, 2),
                    latency=2, allow_tma=True).reshape((TILE_D, TILE_N))

        # Compute attention scores
        qk = ct.full((TILE_M, TILE_N), 0., dtype=np.float32)
        qk = ct.mma(q, k, qk)

        # Apply causal mask
        offs_n = kv_tile_idx * TILE_N + offs_n_tile
        causal_mask = offs_m >= offs_n
        causal_mask = ct.where(causal_mask, 0.0, -np.inf)
        qk = qk + causal_mask

        # ---- Phase 2: Online softmax update ----
        # Compute new max
        qk_max = ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2
        new_max = max(max_val, qk_max)

        # Compute attention weights with numerically stable softmax
        qk_scaled = qk * qk_scale_log2 - new_max
        attn_weights = ct.exp2(qk_scaled, flush_to_zero=True)

        # New sum of exponentials
        new_sum_exp = ct.sum(attn_weights, axis=-1, keepdims=True)

        # Correction factor for previous accumulator
        correction = ct.exp2(max_val - new_max, flush_to_zero=True)

        # Update running statistics
        sum_exp = sum_exp * correction + new_sum_exp
        output = output * correction  # Rescale previous output
        max_val = new_max

        # ---- Phase 3: Load V and accumulate ----
        v = ct.load(V, index=(batch_idx, head_idx, kv_tile_idx, 0),
                    shape=(1, 1, TILE_N, TILE_D),
                    latency=4, allow_tma=True).reshape((TILE_N, TILE_D))

        # Accumulate weighted values
        attn_weights = attn_weights.astype(Q.dtype)
        output = ct.mma(attn_weights, v, output)

    # ===== Final normalization =====
    output = ct.truediv(output, sum_exp, flush_to_zero=True, rounding_mode=RMd.APPROX)
    output = output.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=output)


def cutile_causal_attention_v2(
    q: cp.ndarray,
    k: cp.ndarray,
    v: cp.ndarray,
    n_head: int
) -> cp.ndarray:
    """
    Improved causal attention with better structure.

    Args:
        q: Query tensor (batch, n_head, seq_len, head_dim)
        k: Key tensor (batch, n_head, seq_len, head_dim)
        v: Value tensor (batch, n_head, seq_len, head_dim)
        n_head: Number of attention heads

    Returns:
        Attention output (batch, n_head, seq_len, head_dim)
    """
    if not isinstance(q, cp.ndarray):
        raise ValueError("Tensors must be CuPy arrays on CUDA device")

    batch, n_head, seq_len, head_dim = q.shape

    # Ensure contiguous memory
    if not q.flags.c_contiguous:
        q = cp.ascontiguousarray(q)
    if not k.flags.c_contiguous:
        k = cp.ascontiguousarray(k)
    if not v.flags.c_contiguous:
        v = cp.ascontiguousarray(v)

    qk_scale = 1.0 / math.sqrt(head_dim)
    out = cp.empty_like(q)

    tile_m = 64
    tile_n = 64

    grid_x = math.ceil(seq_len / tile_m)
    grid_y = batch * n_head

    ct.launch(
        cp.cuda.get_current_stream(),
        (grid_x, grid_y, 1),
        causal_attention_kernel_v2,
        (q, k, v, out, qk_scale, head_dim, n_head, tile_m, tile_n)
    )

    return out


# Import original for comparison
def cutile_mha_forward_v2(
    x: cp.ndarray,
    c_attn_weight: cp.ndarray,
    c_attn_bias: cp.ndarray,
    c_proj_weight: cp.ndarray,
    c_proj_bias: cp.ndarray,
    n_head: int,
    c_attn_weight_t: cp.ndarray = None,
    c_proj_weight_t: cp.ndarray = None
) -> cp.ndarray:
    """
    MHA forward with improved attention kernel.
    """
    from .linear import cutile_linear_bias

    batch, seq_len, n_embd = x.shape
    head_dim = n_embd // n_head

    # Combined QKV projection
    qkv = cutile_linear_bias(x, c_attn_weight, c_attn_bias, c_attn_weight_t)

    # Split into Q, K, V
    q, k, v = cp.split(qkv, 3, axis=2)

    # Reshape to (batch, n_head, seq_len, head_dim)
    q = cp.transpose(cp.reshape(q, (batch, seq_len, n_head, head_dim)), (0, 2, 1, 3))
    k = cp.transpose(cp.reshape(k, (batch, seq_len, n_head, head_dim)), (0, 2, 1, 3))
    v = cp.transpose(cp.reshape(v, (batch, seq_len, n_head, head_dim)), (0, 2, 1, 3))

    # Use improved attention kernel
    y = cutile_causal_attention_v2(q, k, v, n_head)

    # Reshape back
    y = cp.transpose(y, (0, 2, 1, 3))
    if not y.flags.c_contiguous:
        y = cp.ascontiguousarray(y)
    y = cp.reshape(y, (batch, seq_len, n_embd))

    # Output projection
    y = cutile_linear_bias(y, c_proj_weight, c_proj_bias, c_proj_weight_t)

    return y


if __name__ == "__main__":
    print("--- Testing Improved Attention Kernel ---\n")

    # Import original for comparison
    from .attention import cutile_causal_attention, cupy_causal_attention

    batch, n_head, seq_len, head_dim = 2, 4, 128, 64

    q = cp.random.randn(batch, n_head, seq_len, head_dim, dtype=cp.float32)
    k = cp.random.randn(batch, n_head, seq_len, head_dim, dtype=cp.float32)
    v = cp.random.randn(batch, n_head, seq_len, head_dim, dtype=cp.float32)

    print(f"Config: batch={batch}, n_head={n_head}, seq_len={seq_len}, head_dim={head_dim}")
    print("Testing correctness...")

    # Test V2
    y_v2 = cutile_causal_attention_v2(q, k, v, n_head)
    y_v1 = cutile_causal_attention(q, k, v, n_head)
    y_ref = cupy_causal_attention(q, k, v, n_head)

    print(f"V2 vs V1 max diff: {cp.abs(y_v2 - y_v1).max():.6f}")
    print(f"V2 vs Reference max diff: {cp.abs(y_v2 - y_ref).max():.6f}")

    cp.testing.assert_allclose(y_v2, y_ref, atol=1e-3, rtol=1e-3)
    print("✓ Correctness test passed!\n")

    # Benchmark
    print("Benchmarking...")
    import time

    # Warmup
    for _ in range(10):
        _ = cutile_causal_attention(q, k, v, n_head)
        _ = cutile_causal_attention_v2(q, k, v, n_head)
    cp.cuda.Stream.null.synchronize()

    # Benchmark V1
    start = time.time()
    for _ in range(100):
        _ = cutile_causal_attention(q, k, v, n_head)
    cp.cuda.Stream.null.synchronize()
    v1_time = (time.time() - start) / 100 * 1000

    # Benchmark V2
    start = time.time()
    for _ in range(100):
        _ = cutile_causal_attention_v2(q, k, v, n_head)
    cp.cuda.Stream.null.synchronize()
    v2_time = (time.time() - start) / 100 * 1000

    print(f"\nResults:")
    print(f"  Original (V1): {v1_time:.3f} ms")
    print(f"  Improved (V2): {v2_time:.3f} ms")

    if v2_time < v1_time:
        speedup = (v1_time - v2_time) / v1_time * 100
        print(f"\n✅ V2 is {speedup:.1f}% faster!")
    else:
        slowdown = (v2_time - v1_time) / v1_time * 100
        print(f"\n⚠️  V2 is {slowdown:.1f}% slower")
        print("   (Code structure changes may not always improve performance)")

    print("\n--- Test complete ---")
