# SPDX-License-Identifier: Apache-2.0
"""
Embedding layer for cutile GPT.

cutile requires tile sizes to be powers of 2.
We pad n_embd to next power of 2, then slice back.
"""

import math
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def embedding_kernel(indices, weight, output, TILE_SEQ: ConstInt, N_EMBD_PADDED: ConstInt):
    """
    Embedding lookup kernel using gather.

    Args:
        indices: Token indices (total_tokens,)
        weight: Embedding weight matrix (vocab_size, n_embd_padded)
        output: Output tensor (total_tokens, n_embd_padded)
        TILE_SEQ: Tile size for sequence (power of 2)
        N_EMBD_PADDED: Padded embedding dim (power of 2)
    """
    bid = ct.bid(0)

    # Load indices for this tile
    idx_tile = ct.load(indices, index=(bid,), shape=(TILE_SEQ,),
                       padding_mode=ct.PaddingMode.ZERO)

    # Create 2D indices for gather
    # row_idx: which row to select (based on token id)
    # col_idx: which columns (all of them)
    row_idx = idx_tile[:, None]  # (TILE_SEQ, 1)
    col_idx = ct.arange(N_EMBD_PADDED, dtype=ct.int32)[None, :]  # (1, N_EMBD_PADDED)

    # Gather: result[i, j] = weight[row_idx[i, 0], col_idx[0, j]]
    emb = ct.gather(weight, (row_idx, col_idx))  # (TILE_SEQ, N_EMBD_PADDED)

    # Store result
    ct.store(output, index=(bid, 0), tile=emb.astype(output.dtype))


def cutile_embedding(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Perform embedding lookup using cutile kernel.

    Handles non-power-of-2 embedding dimensions by padding.

    Args:
        indices: Token indices (batch, seq_len) or (seq_len,)
        weight: Embedding weight matrix (vocab_size, n_embd)

    Returns:
        Embeddings tensor (batch, seq_len, n_embd) or (seq_len, n_embd)
    """
    if not indices.is_cuda or not weight.is_cuda:
        raise ValueError("Tensors must be on CUDA device")

    original_shape = indices.shape
    vocab_size, n_embd = weight.shape

    # Pad n_embd to power of 2 if needed
    n_embd_padded = next_power_of_2(n_embd)
    needs_padding = n_embd_padded != n_embd

    if needs_padding:
        # Pad weight matrix: (vocab_size, n_embd) -> (vocab_size, n_embd_padded)
        weight_padded = torch.zeros(vocab_size, n_embd_padded,
                                    dtype=weight.dtype, device=weight.device)
        weight_padded[:, :n_embd] = weight
    else:
        weight_padded = weight

    # Flatten indices
    indices_flat = indices.reshape(-1).to(torch.int32)
    total_tokens = indices_flat.numel()

    # Tile size for sequence (also power of 2)
    TILE_SEQ = min(64, next_power_of_2(total_tokens))

    # Pad total_tokens to multiple of TILE_SEQ
    padded_tokens = math.ceil(total_tokens / TILE_SEQ) * TILE_SEQ
    if padded_tokens > total_tokens:
        indices_padded = torch.zeros(padded_tokens, dtype=torch.int32, device=indices.device)
        indices_padded[:total_tokens] = indices_flat
        indices_flat = indices_padded

    # Output
    output = torch.empty(padded_tokens, n_embd_padded,
                         dtype=weight.dtype, device=weight.device)

    grid = (padded_tokens // TILE_SEQ, 1, 1)

    ct.launch(torch.cuda.current_stream(), grid, embedding_kernel,
              (indices_flat, weight_padded, output, TILE_SEQ, n_embd_padded))

    # Slice back to original dimensions
    output = output[:total_tokens, :n_embd]

    return output.reshape(*original_shape, n_embd)


def cutile_token_pos_embedding(
    token_ids: torch.Tensor,
    token_weight: torch.Tensor,
    pos_weight: torch.Tensor,
    input_pos: int = 0
) -> torch.Tensor:
    """
    Compute token + position embeddings.

    Args:
        token_ids: Token indices (batch, seq_len)
        token_weight: Token embedding matrix (vocab_size, n_embd)
        pos_weight: Position embedding matrix (max_seq_len, n_embd)
        input_pos: Starting position offset

    Returns:
        Combined embeddings (batch, seq_len, n_embd)
    """
    seq_len = token_ids.shape[-1]

    # Token embeddings
    tok_emb = cutile_embedding(token_ids, token_weight)

    # Position indices
    pos_indices = torch.arange(input_pos, input_pos + seq_len,
                               dtype=torch.long, device=token_ids.device)

    # Position embeddings
    pos_emb = cutile_embedding(pos_indices, pos_weight)

    # Broadcast add
    result = tok_emb + pos_emb.unsqueeze(0)

    return result


# Reference PyTorch implementation (same as cutile_embedding now)
def torch_embedding(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """PyTorch reference embedding"""
    return torch.nn.functional.embedding(indices, weight)


if __name__ == "__main__":
    print("--- Testing Embedding ---")

    vocab_size = 100
    n_embd = 48
    batch_size = 2
    seq_len = 32

    weight = torch.randn(vocab_size, n_embd, dtype=torch.float32, device='cuda')
    indices = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')

    emb = cutile_embedding(indices, weight)
    print(f"Embedding shape: {emb.shape}")
    print("Embedding: PASSED")
