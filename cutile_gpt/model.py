# SPDX-License-Identifier: Apache-2.0
"""
cutile GPT Model

A GPT implementation using cutile kernels for GPU acceleration.
Architecture matches minGPT for weight compatibility.
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from .kernels.gelu import cutile_gelu
    from .kernels.embedding import cutile_embedding
    from .kernels.linear import cutile_linear, cutile_linear_bias
    from .kernels.layernorm import cutile_layer_norm
    from .kernels.attention import cutile_mha_forward
    from .kernels.fused_mlp import cutile_fused_mlp
except ImportError:
    from kernels.gelu import cutile_gelu
    from kernels.embedding import cutile_embedding
    from kernels.linear import cutile_linear, cutile_linear_bias
    from kernels.layernorm import cutile_layer_norm
    from kernels.attention import cutile_mha_forward
    from kernels.fused_mlp import cutile_fused_mlp


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
        self.use_fused_mlp = use_fused_mlp

        # Initialize weight containers
        self.weights = {}
        self._init_weights()

    def _init_weights(self):
        """Initialize weight tensors (random initialization)."""
        cfg = self.config
        n_embd = cfg.n_embd

        # Token and position embeddings
        self.weights['wte'] = torch.randn(
            cfg.vocab_size, n_embd, device=self.device) * 0.02
        self.weights['wpe'] = torch.randn(
            cfg.block_size, n_embd, device=self.device) * 0.02

        # Transformer blocks
        for i in range(cfg.n_layer):
            prefix = f'h.{i}.'

            # LayerNorm 1 (before attention)
            self.weights[prefix + 'ln_1.weight'] = torch.ones(n_embd, device=self.device)
            self.weights[prefix + 'ln_1.bias'] = torch.zeros(n_embd, device=self.device)

            # Attention: c_attn (QKV projection) and c_proj (output projection)
            self.weights[prefix + 'attn.c_attn.weight'] = torch.randn(
                3 * n_embd, n_embd, device=self.device) * 0.02
            self.weights[prefix + 'attn.c_attn.bias'] = torch.zeros(
                3 * n_embd, device=self.device)
            self.weights[prefix + 'attn.c_proj.weight'] = torch.randn(
                n_embd, n_embd, device=self.device) * (0.02 / math.sqrt(2 * cfg.n_layer))
            self.weights[prefix + 'attn.c_proj.bias'] = torch.zeros(
                n_embd, device=self.device)

            # LayerNorm 2 (before MLP)
            self.weights[prefix + 'ln_2.weight'] = torch.ones(n_embd, device=self.device)
            self.weights[prefix + 'ln_2.bias'] = torch.zeros(n_embd, device=self.device)

            # MLP: c_fc (expand) and c_proj (contract)
            self.weights[prefix + 'mlp.c_fc.weight'] = torch.randn(
                4 * n_embd, n_embd, device=self.device) * 0.02
            self.weights[prefix + 'mlp.c_fc.bias'] = torch.zeros(
                4 * n_embd, device=self.device)
            self.weights[prefix + 'mlp.c_proj.weight'] = torch.randn(
                n_embd, 4 * n_embd, device=self.device) * (0.02 / math.sqrt(2 * cfg.n_layer))
            self.weights[prefix + 'mlp.c_proj.bias'] = torch.zeros(
                n_embd, device=self.device)

        # Final LayerNorm
        self.weights['ln_f.weight'] = torch.ones(n_embd, device=self.device)
        self.weights['ln_f.bias'] = torch.zeros(n_embd, device=self.device)

        # Language model head (tied with wte in minGPT, but we keep separate for clarity)
        self.weights['lm_head.weight'] = self.weights['wte']  # Weight tying

    def load_from_mingpt(self, mingpt_model: nn.Module):
        """
        Load weights from a minGPT model.

        Args:
            mingpt_model: A minGPT GPT model instance
        """
        sd = mingpt_model.state_dict()

        # Token and position embeddings
        self.weights['wte'] = sd['transformer.wte.weight'].to(self.device)
        self.weights['wpe'] = sd['transformer.wpe.weight'].to(self.device)

        # Transformer blocks
        for i in range(self.config.n_layer):
            prefix = f'h.{i}.'
            sd_prefix = f'transformer.h.{i}.'

            # LayerNorm 1
            self.weights[prefix + 'ln_1.weight'] = sd[sd_prefix + 'ln_1.weight'].to(self.device)
            self.weights[prefix + 'ln_1.bias'] = sd[sd_prefix + 'ln_1.bias'].to(self.device)

            # Attention
            self.weights[prefix + 'attn.c_attn.weight'] = sd[sd_prefix + 'attn.c_attn.weight'].to(self.device)
            self.weights[prefix + 'attn.c_attn.bias'] = sd[sd_prefix + 'attn.c_attn.bias'].to(self.device)
            self.weights[prefix + 'attn.c_proj.weight'] = sd[sd_prefix + 'attn.c_proj.weight'].to(self.device)
            self.weights[prefix + 'attn.c_proj.bias'] = sd[sd_prefix + 'attn.c_proj.bias'].to(self.device)

            # LayerNorm 2
            self.weights[prefix + 'ln_2.weight'] = sd[sd_prefix + 'ln_2.weight'].to(self.device)
            self.weights[prefix + 'ln_2.bias'] = sd[sd_prefix + 'ln_2.bias'].to(self.device)

            # MLP
            self.weights[prefix + 'mlp.c_fc.weight'] = sd[sd_prefix + 'mlp.c_fc.weight'].to(self.device)
            self.weights[prefix + 'mlp.c_fc.bias'] = sd[sd_prefix + 'mlp.c_fc.bias'].to(self.device)
            self.weights[prefix + 'mlp.c_proj.weight'] = sd[sd_prefix + 'mlp.c_proj.weight'].to(self.device)
            self.weights[prefix + 'mlp.c_proj.bias'] = sd[sd_prefix + 'mlp.c_proj.bias'].to(self.device)

        # Final LayerNorm
        self.weights['ln_f.weight'] = sd['transformer.ln_f.weight'].to(self.device)
        self.weights['ln_f.bias'] = sd['transformer.ln_f.bias'].to(self.device)

        # LM head (weight tied with wte)
        self.weights['lm_head.weight'] = sd['lm_head.weight'].to(self.device)

    def __call__(self, idx: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Make model callable."""
        return self.forward(idx)

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, None]:
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
        pos = torch.arange(0, seq_len, dtype=torch.long, device=self.device)
        pos_emb = cutile_embedding(pos, self.weights['wpe'])

        # Combine embeddings
        x = tok_emb + pos_emb.unsqueeze(0)

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
                cfg.n_head
            )
            x = x + attn_out

            # Pre-LayerNorm + MLP + Residual
            x_norm = cutile_layer_norm(
                x,
                self.weights[prefix + 'ln_2.weight'],
                self.weights[prefix + 'ln_2.bias']
            )

            if self.use_fused_mlp:
                # Fused MLP: single kernel for Linear -> GELU -> Linear
                mlp_out = cutile_fused_mlp(
                    x_norm,
                    self.weights[prefix + 'mlp.c_fc.weight'],
                    self.weights[prefix + 'mlp.c_fc.bias'],
                    self.weights[prefix + 'mlp.c_proj.weight'],
                    self.weights[prefix + 'mlp.c_proj.bias']
                )
            else:
                # Separate kernels: Linear -> GELU -> Linear
                hidden = cutile_linear_bias(
                    x_norm,
                    self.weights[prefix + 'mlp.c_fc.weight'],
                    self.weights[prefix + 'mlp.c_fc.bias']
                )
                hidden = cutile_gelu(hidden)
                mlp_out = cutile_linear_bias(
                    hidden,
                    self.weights[prefix + 'mlp.c_proj.weight'],
                    self.weights[prefix + 'mlp.c_proj.bias']
                )
            x = x + mlp_out

        # Final LayerNorm
        x = cutile_layer_norm(
            x,
            self.weights['ln_f.weight'],
            self.weights['ln_f.bias']
        )

        # Language model head
        logits = cutile_linear(x, self.weights['lm_head.weight'])

        return logits, None

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
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
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self.forward(idx_cond)

            # Get logits for last position and scale by temperature
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat([idx, idx_next], dim=1)

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
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), device='cuda')

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
    print(f"Generated tokens: {generated[0].tolist()}")

    print("\n--- All CutileGPT tests passed! ---")
