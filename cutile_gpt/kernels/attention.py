# SPDX-License-Identifier: Apache-2.0
"""
Causal Self-Attention for cutile GPT.

Optimized with:
- num_ctas and occupancy hints
- latency hints for memory access
- flush_to_zero and approx rounding for perf
- Larger tile sizes for better occupancy
"""

import math
import torch
import cuda.tile as ct
from cuda.tile import RoundingMode as RMd
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

INV_LOG_2 = 1.0 / math.log(2)


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def causal_attention_kernel(
    Q, K, V, Out,
    qk_scale: float,
    TILE_D: ConstInt,
    N_HEAD: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt
):
    """
    Optimized causal multi-head self-attention kernel.

    Args:
        Q: Query tensor (batch, n_head, seq_len, head_dim)
        K: Key tensor (batch, n_head, seq_len, head_dim)
        V: Value tensor (batch, n_head, seq_len, head_dim)
        Out: Output tensor (batch, n_head, seq_len, head_dim)
        qk_scale: Scale factor (1/sqrt(head_dim))
        TILE_D: Head dimension
        N_HEAD: Number of attention heads
        TILE_M: Tile size for query sequence
        TILE_N: Tile size for key/value sequence
    """
    bid_x = ct.bid(0)  # Query tile index
    bid_y = ct.bid(1)  # Batch * head index

    batch_idx = bid_y // N_HEAD
    head_idx = bid_y % N_HEAD

    # Scale for exp2 optimization
    qk_scale_log2 = qk_scale * INV_LOG_2

    # Query position offsets
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=np.int32)
    offs_m = offs_m[:, None]

    # Key/Value position offsets
    offs_n_tile = ct.arange(TILE_N, dtype=np.int32)
    offs_n_tile = offs_n_tile[None, :]

    # Online softmax accumulators
    m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)

    # Load query tile with latency hint and TMA
    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0),
                shape=(1, 1, TILE_M, TILE_D), latency=4, allow_tma=True).reshape((TILE_M, TILE_D))

    # Causal masking: only attend to positions <= current position
    seq_len = K.shape[2]
    m_end = (bid_x + 1) * TILE_M
    Tc = ct.cdiv(min(m_end, seq_len), TILE_N)

    # Loop over K, V blocks
    for j in range(0, Tc):
        # Load K tile (transposed for matmul) with latency hint and TMA
        k = ct.load(K, index=(batch_idx, head_idx, 0, j),
                    shape=(1, 1, TILE_D, TILE_N),
                    order=(0, 1, 3, 2),
                    latency=2, allow_tma=True).reshape((TILE_D, TILE_N))

        # QK^T
        qk = ct.full((TILE_M, TILE_N), 0., dtype=np.float32)
        qk = ct.mma(q, k, qk)

        # Apply causal mask
        offs_n = j * TILE_N + offs_n_tile
        mask = offs_m >= offs_n
        mask = ct.where(mask, 0.0, -np.inf)
        qk += mask

        # Online softmax
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2)
        qk = qk * qk_scale_log2 - m_ij
        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # Load V and accumulate with latency hint and TMA
        v = ct.load(V, index=(batch_idx, head_idx, j, 0),
                    shape=(1, 1, TILE_N, TILE_D),
                    latency=4, allow_tma=True).reshape((TILE_N, TILE_D))
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    # Final normalization with approximate division for performance
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


def cutile_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_head: int
) -> torch.Tensor:
    """
    Compute causal multi-head self-attention.

    This function expects Q, K, V already projected and reshaped to
    (batch, n_head, seq_len, head_dim).

    Args:
        q: Query tensor (batch, n_head, seq_len, head_dim)
        k: Key tensor (batch, n_head, seq_len, head_dim)
        v: Value tensor (batch, n_head, seq_len, head_dim)
        n_head: Number of attention heads

    Returns:
        Attention output (batch, n_head, seq_len, head_dim)
    """
    if not q.is_cuda:
        raise ValueError("Tensors must be on CUDA device")

    batch, n_head, seq_len, head_dim = q.shape

    # Ensure contiguous memory layout for better performance
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Scale factor
    qk_scale = 1.0 / math.sqrt(head_dim)

    # Output tensor
    out = torch.empty_like(q)

    # Larger tile sizes for better occupancy (like official sample)
    # Use 128 for longer sequences, 64 for shorter
    tile_m = min(128, seq_len) if seq_len >= 128 else min(64, seq_len)
    tile_n = min(128, seq_len) if seq_len >= 128 else min(64, seq_len)

    # Grid dimensions
    grid_x = math.ceil(seq_len / tile_m)
    grid_y = batch * n_head

    ct.launch(
        torch.cuda.current_stream(),
        (grid_x, grid_y, 1),
        causal_attention_kernel,
        (q, k, v, out, qk_scale, head_dim, n_head, tile_m, tile_n)
    )

    return out


def cutile_mha_forward(
    x: torch.Tensor,
    c_attn_weight: torch.Tensor,
    c_attn_bias: torch.Tensor,
    c_proj_weight: torch.Tensor,
    c_proj_bias: torch.Tensor,
    n_head: int
) -> torch.Tensor:
    """
    Full multi-head attention forward pass (matching minGPT).

    Args:
        x: Input tensor (batch, seq_len, n_embd)
        c_attn_weight: Combined QKV projection weight (3*n_embd, n_embd)
        c_attn_bias: Combined QKV projection bias (3*n_embd,)
        c_proj_weight: Output projection weight (n_embd, n_embd)
        c_proj_bias: Output projection bias (n_embd,)
        n_head: Number of attention heads

    Returns:
        Output tensor (batch, seq_len, n_embd)
    """
    from .linear import cutile_linear_bias

    batch, seq_len, n_embd = x.shape
    head_dim = n_embd // n_head

    # Combined QKV projection
    qkv = cutile_linear_bias(x, c_attn_weight, c_attn_bias)  # (B, T, 3*n_embd)

    # Split into Q, K, V
    q, k, v = qkv.split(n_embd, dim=2)

    # Reshape to (batch, n_head, seq_len, head_dim)
    q = q.view(batch, seq_len, n_head, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, n_head, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, n_head, head_dim).transpose(1, 2)

    # Attention
    y = cutile_causal_attention(q, k, v, n_head)

    # Reshape back: (batch, n_head, seq_len, head_dim) -> (batch, seq_len, n_embd)
    y = y.transpose(1, 2).contiguous().view(batch, seq_len, n_embd)

    # Output projection
    y = cutile_linear_bias(y, c_proj_weight, c_proj_bias)

    return y


# Reference PyTorch implementation
def torch_causal_attention(q, k, v, n_head):
    """PyTorch reference causal attention"""
    batch, n_head, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    # QK^T
    att = (q @ k.transpose(-2, -1)) * scale

    # Causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
    att = att.masked_fill(mask == 0, float('-inf'))

    # Softmax
    att = torch.softmax(att, dim=-1)

    # Weighted sum
    y = att @ v
    return y


if __name__ == "__main__":
    print("--- Testing cutile Causal Attention kernel ---")

    batch, n_head, seq_len, head_dim = 2, 3, 32, 16
    n_embd = n_head * head_dim

    q = torch.randn(batch, n_head, seq_len, head_dim, dtype=torch.float32, device='cuda')
    k = torch.randn(batch, n_head, seq_len, head_dim, dtype=torch.float32, device='cuda')
    v = torch.randn(batch, n_head, seq_len, head_dim, dtype=torch.float32, device='cuda')

    y_cutile = cutile_causal_attention(q, k, v, n_head)
    y_torch = torch_causal_attention(q, k, v, n_head)

    print(f"Input Q shape: {q.shape}")
    print(f"Output shape: {y_cutile.shape}")
    print(f"Max diff: {(y_cutile - y_torch).abs().max().item():.6f}")

    torch.testing.assert_close(y_cutile, y_torch, atol=1e-3, rtol=1e-3)
    print("Causal attention test passed!")

    print("\n--- All Attention tests passed! ---")
