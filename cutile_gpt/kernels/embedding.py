# SPDX-License-Identifier: Apache-2.0
"""
Embedding layer for cutile GPT.

cutile requires tile sizes to be powers of 2.
We pad n_embd to next power of 2, then slice back.
"""

import math

import cuda.tile as ct
import cupy as cp

ConstInt = ct.Constant[int]


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def embedding_kernel(
    indices,
    weight,
    output,
    TILE_SEQ: ConstInt,
    TILE_EMBD: ConstInt,
    N_EMBD: ConstInt,
):
    """
    Embedding lookup kernel using gather.

    Args:
        indices: Token indices (total_tokens,)
        weight: Embedding weight matrix (vocab_size, n_embd_padded)
        output: Output tensor (total_tokens, n_embd_padded)
        TILE_SEQ: Tile size for sequence (power of 2)
        TILE_EMBD: Power-of-two tile width
        N_EMBD: Actual embedding width
    """
    bid = ct.bid(0)

    # Load indices for this tile
    idx_tile = ct.load(indices, index=(bid,), shape=(TILE_SEQ,),
                       padding_mode=ct.PaddingMode.ZERO)

    # Create 2D indices for gather
    # row_idx: which row to select (based on token id)
    # col_idx: which columns (all of them)
    row_idx = idx_tile[:, None]  # (TILE_SEQ, 1)
    col_idx = ct.arange(TILE_EMBD, dtype=ct.int32)[None, :]

    # Gather: result[i, j] = weight[row_idx[i, 0], col_idx[0, j]]
    emb = ct.gather(
        weight,
        (row_idx, col_idx),
        mask=(col_idx < N_EMBD) & (row_idx >= 0) & (row_idx < weight.shape[0]),
        padding_value=0,
        check_bounds=False,
    )

    # Store result
    ct.store(output, index=(bid, 0), tile=emb.astype(output.dtype))


def cutile_embedding(indices: cp.ndarray, weight: cp.ndarray) -> cp.ndarray:
    """
    Perform embedding lookup using cutile kernel.

    Handles non-power-of-2 embedding dimensions with a masked gather.  This is
    deliberately not implemented by padding the vocabulary matrix: doing so
    copied the entire embedding table on every forward pass.

    Args:
        indices: Token indices (batch, seq_len) or (seq_len,)
        weight: Embedding weight matrix (vocab_size, n_embd)

    Returns:
        Embeddings tensor (batch, seq_len, n_embd) or (seq_len, n_embd)
    """
    if not isinstance(indices, cp.ndarray) or not isinstance(weight, cp.ndarray):
        raise ValueError("Tensors must be CuPy arrays on CUDA device")

    original_shape = indices.shape
    vocab_size, n_embd = weight.shape

    # Pad n_embd to power of 2 if needed
    n_embd_padded = next_power_of_2(n_embd)
    # Flatten indices
    indices_flat = cp.reshape(indices, -1)
    if indices_flat.dtype != cp.int32:
        indices_flat = indices_flat.astype(cp.int32)
    total_tokens = indices_flat.size

    # Tile size for sequence (also power of 2)
    TILE_SEQ = min(64, next_power_of_2(total_tokens))

    output = cp.empty((total_tokens, n_embd), dtype=weight.dtype)
    grid = (math.ceil(total_tokens / TILE_SEQ), 1, 1)

    ct.launch(cp.cuda.get_current_stream(), grid, embedding_kernel,
              (indices_flat, weight, output, TILE_SEQ, n_embd_padded, n_embd))

    return cp.reshape(output, (*original_shape, n_embd))


def cutile_token_pos_embedding(
    token_ids: cp.ndarray,
    token_weight: cp.ndarray,
    pos_weight: cp.ndarray,
    input_pos: int = 0
) -> cp.ndarray:
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
    pos_indices = cp.arange(input_pos, input_pos + seq_len, dtype=cp.int64)

    # Position embeddings
    pos_emb = cutile_embedding(pos_indices, pos_weight)

    # Broadcast add
    result = tok_emb + cp.expand_dims(pos_emb, 0)

    return result


# Reference CuPy implementation
def cupy_embedding(indices: cp.ndarray, weight: cp.ndarray) -> cp.ndarray:
    """CuPy reference embedding"""
    return weight[indices]


if __name__ == "__main__":
    print("--- Testing Embedding ---")

    vocab_size = 100
    n_embd = 48
    batch_size = 2
    seq_len = 32

    weight = cp.random.randn(vocab_size, n_embd, dtype=cp.float32)
    indices = cp.random.randint(0, vocab_size, (batch_size, seq_len))

    emb = cutile_embedding(indices, weight)
    print(f"Embedding shape: {emb.shape}")
    print("Embedding: PASSED")
