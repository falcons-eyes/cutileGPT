# SPDX-License-Identifier: Apache-2.0
"""
cutile GPT Model

A GPT implementation using cutile kernels for GPU acceleration.
Architecture matches minGPT for weight compatibility.
"""

import math
import cupy as cp
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from .kernels.gelu import cutile_gelu
    from .kernels.embedding import cutile_embedding
    from .kernels.linear import cutile_linear, cutile_linear_bias
    from .kernels.layernorm import cutile_layer_norm
    from .kernels.attention import cutile_mha_forward
except ImportError:
    from kernels.gelu import cutile_gelu
    from kernels.embedding import cutile_embedding
    from kernels.linear import cutile_linear, cutile_linear_bias
    from kernels.layernorm import cutile_layer_norm
    from kernels.attention import cutile_mha_forward


@dataclass
class CutileGPTConfig:
    """Configuration for CutileGPT model."""
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    # Predefined configurations (matching minGPT)
    @classmethod
    def gpt_nano(cls) -> 'CutileGPTConfig':
        """Smallest config for testing: 3 layers, 48 dims, 3 heads"""
        return cls(n_layer=3, n_head=3, n_embd=48, block_size=128)

    @classmethod
    def gpt_micro(cls) -> 'CutileGPTConfig':
        """Micro config: 4 layers, 128 dims, 4 heads"""
        return cls(n_layer=4, n_head=4, n_embd=128, block_size=256)

    @classmethod
    def gpt_mini(cls) -> 'CutileGPTConfig':
        """Mini config: 6 layers, 192 dims, 6 heads"""
        return cls(n_layer=6, n_head=6, n_embd=192, block_size=256)

    @classmethod
    def gpt2(cls) -> 'CutileGPTConfig':
        """GPT-2 (124M params): 12 layers, 768 dims, 12 heads"""
        return cls(n_layer=12, n_head=12, n_embd=768, block_size=1024)

    # Tile-optimized configs (power of 2 dimensions = no padding overhead)
    @classmethod
    def gpt_tile_small(cls) -> 'CutileGPTConfig':
        """Tile-optimized small: 4 layers, 64 dims, 4 heads (no padding)"""
        return cls(n_layer=4, n_head=4, n_embd=64, block_size=128)

    @classmethod
    def gpt_tile_medium(cls) -> 'CutileGPTConfig':
        """Tile-optimized medium: 6 layers, 128 dims, 4 heads (no padding)"""
        return cls(n_layer=6, n_head=4, n_embd=128, block_size=256)

    @classmethod
    def gpt_tile_large(cls) -> 'CutileGPTConfig':
        """Tile-optimized large: 8 layers, 256 dims, 8 heads (no padding)"""
        return cls(n_layer=8, n_head=8, n_embd=256, block_size=512)


class CutileGPT:
    """
    GPT model using cutile kernels for inference.

    This class holds the model weights and provides forward pass
    using cutile CUDA kernels.
    """

    def __init__(self, config: CutileGPTConfig, device: str = 'cuda', use_fused_mlp: bool = False):
        self.config = config
        self.device = device
        # use_fused_mlp is deprecated but kept for backward compatibility
        self.use_fused_mlp = False  # Always False, fused MLP removed

        # Initialize weight containers
        self.weights = {}
        self.weight_transposes = {}  # Pre-computed transposes for performance
        self._init_weights()
        self._precompute_transposes()

    def _init_weights(self):
        """Initialize weight tensors (random initialization)."""
        cfg = self.config
        n_embd = cfg.n_embd

        # Token and position embeddings
        self.weights['wte'] = cp.random.randn(cfg.vocab_size, n_embd, dtype=cp.float32) * 0.02
        self.weights['wpe'] = cp.random.randn(cfg.block_size, n_embd, dtype=cp.float32) * 0.02

        # Transformer blocks
        for i in range(cfg.n_layer):
            prefix = f'h.{i}.'

            # LayerNorm 1 (before attention)
            self.weights[prefix + 'ln_1.weight'] = cp.ones(n_embd, dtype=cp.float32)
            self.weights[prefix + 'ln_1.bias'] = cp.zeros(n_embd, dtype=cp.float32)

            # Attention: c_attn (QKV projection) and c_proj (output projection)
            self.weights[prefix + 'attn.c_attn.weight'] = cp.random.randn(
                3 * n_embd, n_embd, dtype=cp.float32) * 0.02
            self.weights[prefix + 'attn.c_attn.bias'] = cp.zeros(3 * n_embd, dtype=cp.float32)
            self.weights[prefix + 'attn.c_proj.weight'] = cp.random.randn(
                n_embd, n_embd, dtype=cp.float32) * (0.02 / math.sqrt(2 * cfg.n_layer))
            self.weights[prefix + 'attn.c_proj.bias'] = cp.zeros(n_embd, dtype=cp.float32)

            # LayerNorm 2 (before MLP)
            self.weights[prefix + 'ln_2.weight'] = cp.ones(n_embd, dtype=cp.float32)
            self.weights[prefix + 'ln_2.bias'] = cp.zeros(n_embd, dtype=cp.float32)

            # MLP: c_fc (expand) and c_proj (contract)
            self.weights[prefix + 'mlp.c_fc.weight'] = cp.random.randn(
                4 * n_embd, n_embd, dtype=cp.float32) * 0.02
            self.weights[prefix + 'mlp.c_fc.bias'] = cp.zeros(4 * n_embd, dtype=cp.float32)
            self.weights[prefix + 'mlp.c_proj.weight'] = cp.random.randn(
                n_embd, 4 * n_embd, dtype=cp.float32) * (0.02 / math.sqrt(2 * cfg.n_layer))
            self.weights[prefix + 'mlp.c_proj.bias'] = cp.zeros(n_embd, dtype=cp.float32)

        # Final LayerNorm
        self.weights['ln_f.weight'] = cp.ones(n_embd, dtype=cp.float32)
        self.weights['ln_f.bias'] = cp.zeros(n_embd, dtype=cp.float32)

        # Language model head (tied with wte in minGPT, but we keep separate for clarity)
        self.weights['lm_head.weight'] = self.weights['wte']  # Weight tying

    def _precompute_transposes(self):
        """Precompute and cache weight transposes for performance optimization."""
        for key, weight in self.weights.items():
            # Only transpose 2D weight matrices (not biases or embeddings)
            if 'weight' in key and weight.ndim == 2:
                weight_t = cp.transpose(weight)
                if not weight_t.flags.c_contiguous:
                    weight_t = cp.ascontiguousarray(weight_t)
                self.weight_transposes[key] = weight_t

    def load_from_mingpt(self, mingpt_model):
        """
        Load weights from a minGPT model (PyTorch).

        Args:
            mingpt_model: A minGPT GPT model instance
        """
        import torch
        sd = mingpt_model.state_dict()

        # Convert PyTorch tensors to CuPy arrays
        def torch_to_cupy(t):
            return cp.asarray(t.detach().cpu().numpy())

        # Token and position embeddings
        self.weights['wte'] = torch_to_cupy(sd['transformer.wte.weight'])
        self.weights['wpe'] = torch_to_cupy(sd['transformer.wpe.weight'])

        # Transformer blocks
        for i in range(self.config.n_layer):
            prefix = f'h.{i}.'
            sd_prefix = f'transformer.h.{i}.'

            # LayerNorm 1
            self.weights[prefix + 'ln_1.weight'] = torch_to_cupy(sd[sd_prefix + 'ln_1.weight'])
            self.weights[prefix + 'ln_1.bias'] = torch_to_cupy(sd[sd_prefix + 'ln_1.bias'])

            # Attention
            self.weights[prefix + 'attn.c_attn.weight'] = torch_to_cupy(sd[sd_prefix + 'attn.c_attn.weight'])
            self.weights[prefix + 'attn.c_attn.bias'] = torch_to_cupy(sd[sd_prefix + 'attn.c_attn.bias'])
            self.weights[prefix + 'attn.c_proj.weight'] = torch_to_cupy(sd[sd_prefix + 'attn.c_proj.weight'])
            self.weights[prefix + 'attn.c_proj.bias'] = torch_to_cupy(sd[sd_prefix + 'attn.c_proj.bias'])

            # LayerNorm 2
            self.weights[prefix + 'ln_2.weight'] = torch_to_cupy(sd[sd_prefix + 'ln_2.weight'])
            self.weights[prefix + 'ln_2.bias'] = torch_to_cupy(sd[sd_prefix + 'ln_2.bias'])

            # MLP
            self.weights[prefix + 'mlp.c_fc.weight'] = torch_to_cupy(sd[sd_prefix + 'mlp.c_fc.weight'])
            self.weights[prefix + 'mlp.c_fc.bias'] = torch_to_cupy(sd[sd_prefix + 'mlp.c_fc.bias'])
            self.weights[prefix + 'mlp.c_proj.weight'] = torch_to_cupy(sd[sd_prefix + 'mlp.c_proj.weight'])
            self.weights[prefix + 'mlp.c_proj.bias'] = torch_to_cupy(sd[sd_prefix + 'mlp.c_proj.bias'])

        # Final LayerNorm
        self.weights['ln_f.weight'] = torch_to_cupy(sd['transformer.ln_f.weight'])
        self.weights['ln_f.bias'] = torch_to_cupy(sd['transformer.ln_f.bias'])

        # LM head (weight tied with wte)
        self.weights['lm_head.weight'] = torch_to_cupy(sd['lm_head.weight'])

        # Recompute transposes after loading new weights
        self._precompute_transposes()

    def __call__(self, idx: cp.ndarray) -> Tuple[cp.ndarray, None]:
        """Make model callable."""
        return self.forward(idx)

    def forward(self, idx: cp.ndarray) -> Tuple[cp.ndarray, None]:
        """
        Forward pass using cutile kernels.

        Args:
            idx: Token indices (batch, seq_len)

        Returns:
            Tuple of (logits, None) - None is for loss (not computed in inference)
        """
        batch_size, seq_len = idx.shape
        cfg = self.config
        assert seq_len <= cfg.block_size, f"Sequence length {seq_len} > block_size {cfg.block_size}"

        # Token embeddings
        tok_emb = cutile_embedding(idx, self.weights['wte'])

        # Position embeddings
        pos = cp.arange(0, seq_len, dtype=cp.int64)
        pos_emb = cutile_embedding(pos, self.weights['wpe'])

        # Combine embeddings
        x = tok_emb + cp.expand_dims(pos_emb, 0)

        # Transformer blocks
        for i in range(cfg.n_layer):
            prefix = f'h.{i}.'

            # Pre-LayerNorm + Attention + Residual
            x_norm = cutile_layer_norm(
                x,
                self.weights[prefix + 'ln_1.weight'],
                self.weights[prefix + 'ln_1.bias']
            )
            attn_out = cutile_mha_forward(
                x_norm,
                self.weights[prefix + 'attn.c_attn.weight'],
                self.weights[prefix + 'attn.c_attn.bias'],
                self.weights[prefix + 'attn.c_proj.weight'],
                self.weights[prefix + 'attn.c_proj.bias'],
                cfg.n_head,
                self.weight_transposes.get(prefix + 'attn.c_attn.weight'),
                self.weight_transposes.get(prefix + 'attn.c_proj.weight')
            )
            x = x + attn_out

            # Pre-LayerNorm + MLP + Residual
            x_norm = cutile_layer_norm(
                x,
                self.weights[prefix + 'ln_2.weight'],
                self.weights[prefix + 'ln_2.bias']
            )

            # MLP: Linear -> GELU -> Linear
            hidden = cutile_linear_bias(
                x_norm,
                self.weights[prefix + 'mlp.c_fc.weight'],
                self.weights[prefix + 'mlp.c_fc.bias'],
                self.weight_transposes.get(prefix + 'mlp.c_fc.weight')
            )
            hidden = cutile_gelu(hidden)
            mlp_out = cutile_linear_bias(
                hidden,
                self.weights[prefix + 'mlp.c_proj.weight'],
                self.weights[prefix + 'mlp.c_proj.bias'],
                self.weight_transposes.get(prefix + 'mlp.c_proj.weight')
            )
            x = x + mlp_out

        # Final LayerNorm
        x = cutile_layer_norm(
            x,
            self.weights['ln_f.weight'],
            self.weights['ln_f.bias']
        )

        # Language model head
        logits = cutile_linear(x, self.weights['lm_head.weight'],
                              self.weight_transposes.get('lm_head.weight'))

        return logits, None

    def generate(
        self,
        idx: cp.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> cp.ndarray:
        """
        Autoregressive generation.

        Args:
            idx: Initial token indices (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens

        Returns:
            Extended token sequence (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self.forward(idx_cond)

            # Get logits for last position and scale by temperature
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                # Get top k values and indices
                k = min(top_k, logits.shape[-1])
                # Use partition to get kth smallest element
                top_vals = cp.partition(logits, -k, axis=-1)[:, -k:]
                kth_val = cp.min(top_vals, axis=-1, keepdims=True)
                logits = cp.where(logits < kth_val, float('-inf'), logits)

            # Softmax
            logits_max = cp.max(logits, axis=-1, keepdims=True)
            exp_logits = cp.exp(logits - logits_max)
            probs = exp_logits / cp.sum(exp_logits, axis=-1, keepdims=True)

            # Sample using CuPy's multinomial-like sampling
            # CuPy doesn't have multinomial, so we use cumsum + searchsorted
            batch_size = probs.shape[0]
            idx_next = cp.zeros((batch_size, 1), dtype=cp.int64)

            for b in range(batch_size):
                cumsum_probs = cp.cumsum(probs[b])
                rand_val = cp.random.rand().astype(cp.float32)
                idx_next[b, 0] = cp.searchsorted(cumsum_probs, rand_val)

            # Append
            idx = cp.concatenate([idx, idx_next], axis=1)

        return idx


if __name__ == "__main__":
    print("--- Testing CutileGPT Model ---")

    # Create gpt-nano config
    config = CutileGPTConfig.gpt_nano()
    config.vocab_size = 100  # Small vocab for testing
    config.block_size = 64

    print(f"Config: n_layer={config.n_layer}, n_head={config.n_head}, "
          f"n_embd={config.n_embd}, vocab_size={config.vocab_size}")

    # Create model
    model = CutileGPT(config)

    # Test forward pass
    batch_size = 2
    seq_len = 32
    idx = cp.random.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\nInput shape: {idx.shape}")

    logits, _ = model.forward(idx)
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {config.vocab_size})")

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    print("\nForward pass test passed!")

    # Test generation
    print("\n--- Testing Generation ---")
    generated = model.generate(idx[:1, :5], max_new_tokens=10)
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated tokens: {cp.asnumpy(generated[0]).tolist()}")

    print("\n--- All CutileGPT tests passed! ---")
