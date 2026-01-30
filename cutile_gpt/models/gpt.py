# SPDX-License-Identifier: Apache-2.0
"""
GPT Model Implementation using Tile Programming

A unified GPT implementation using cutile kernels for GPU acceleration.
Supports loading weights from HuggingFace and minGPT.
"""

import math
import cupy as cp
from typing import Optional, Tuple, Dict

from .config import GPTConfig
from ..kernels import (
    cutile_gelu,
    cutile_embedding,
    cutile_linear,
    cutile_linear_bias,
    cutile_layer_norm,
    cutile_causal_attention,
)


class CutileGPT:
    """
    GPT model using cutile kernels for inference.

    This class holds the model weights and provides forward pass
    using cutile CUDA kernels.
    """

    def __init__(self, config: GPTConfig):
        self.config = config
        self.weights: Dict[str, cp.ndarray] = {}
        self.weight_transposes: Dict[str, cp.ndarray] = {}
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

            # LayerNorm 1
            self.weights[prefix + 'ln_1.weight'] = cp.ones(n_embd, dtype=cp.float32)
            self.weights[prefix + 'ln_1.bias'] = cp.zeros(n_embd, dtype=cp.float32)

            # Attention
            self.weights[prefix + 'attn.c_attn.weight'] = cp.random.randn(
                3 * n_embd, n_embd, dtype=cp.float32) * 0.02
            self.weights[prefix + 'attn.c_attn.bias'] = cp.zeros(3 * n_embd, dtype=cp.float32)
            self.weights[prefix + 'attn.c_proj.weight'] = cp.random.randn(
                n_embd, n_embd, dtype=cp.float32) * (0.02 / math.sqrt(2 * cfg.n_layer))
            self.weights[prefix + 'attn.c_proj.bias'] = cp.zeros(n_embd, dtype=cp.float32)

            # LayerNorm 2
            self.weights[prefix + 'ln_2.weight'] = cp.ones(n_embd, dtype=cp.float32)
            self.weights[prefix + 'ln_2.bias'] = cp.zeros(n_embd, dtype=cp.float32)

            # MLP
            self.weights[prefix + 'mlp.c_fc.weight'] = cp.random.randn(
                4 * n_embd, n_embd, dtype=cp.float32) * 0.02
            self.weights[prefix + 'mlp.c_fc.bias'] = cp.zeros(4 * n_embd, dtype=cp.float32)
            self.weights[prefix + 'mlp.c_proj.weight'] = cp.random.randn(
                n_embd, 4 * n_embd, dtype=cp.float32) * (0.02 / math.sqrt(2 * cfg.n_layer))
            self.weights[prefix + 'mlp.c_proj.bias'] = cp.zeros(n_embd, dtype=cp.float32)

        # Final LayerNorm
        self.weights['ln_f.weight'] = cp.ones(n_embd, dtype=cp.float32)
        self.weights['ln_f.bias'] = cp.zeros(n_embd, dtype=cp.float32)

        # LM head (tied with wte)
        self.weights['lm_head.weight'] = self.weights['wte']

    def _precompute_transposes(self):
        """Precompute and cache weight transposes for performance optimization."""
        for key, weight in self.weights.items():
            if 'weight' in key and weight.ndim == 2:
                weight_t = cp.transpose(weight)
                if not weight_t.flags.c_contiguous:
                    weight_t = cp.ascontiguousarray(weight_t)
                self.weight_transposes[key] = weight_t

    def __call__(self, idx: cp.ndarray) -> Tuple[cp.ndarray, None]:
        return self.forward(idx)

    def forward(self, idx: cp.ndarray) -> Tuple[cp.ndarray, None]:
        """
        Forward pass using cutile kernels.

        Args:
            idx: Token indices (batch, seq_len)

        Returns:
            Tuple of (logits, None)
        """
        batch_size, seq_len = idx.shape
        cfg = self.config
        assert seq_len <= cfg.block_size, f"Sequence length {seq_len} > block_size {cfg.block_size}"

        # Embeddings
        tok_emb = cutile_embedding(idx, self.weights['wte'])
        pos = cp.arange(0, seq_len, dtype=cp.int64)
        pos_emb = cutile_embedding(pos, self.weights['wpe'])
        x = tok_emb + cp.expand_dims(pos_emb, 0)

        # Transformer blocks
        for i in range(cfg.n_layer):
            prefix = f'h.{i}.'

            # Attention
            x_norm = cutile_layer_norm(
                x, self.weights[prefix + 'ln_1.weight'], self.weights[prefix + 'ln_1.bias']
            )

            # QKV projection
            qkv = cutile_linear_bias(
                x_norm,
                self.weights[prefix + 'attn.c_attn.weight'],
                self.weights[prefix + 'attn.c_attn.bias'],
                self.weight_transposes.get(prefix + 'attn.c_attn.weight')
            )

            # Split and reshape for attention
            q, k, v = cp.split(qkv, 3, axis=2)
            head_dim = cfg.n_embd // cfg.n_head
            q = cp.transpose(cp.reshape(q, (batch_size, seq_len, cfg.n_head, head_dim)), (0, 2, 1, 3))
            k = cp.transpose(cp.reshape(k, (batch_size, seq_len, cfg.n_head, head_dim)), (0, 2, 1, 3))
            v = cp.transpose(cp.reshape(v, (batch_size, seq_len, cfg.n_head, head_dim)), (0, 2, 1, 3))

            # Attention
            attn_out = cutile_causal_attention(q, k, v, cfg.n_head)

            # Reshape back
            attn_out = cp.transpose(attn_out, (0, 2, 1, 3))
            if not attn_out.flags.c_contiguous:
                attn_out = cp.ascontiguousarray(attn_out)
            attn_out = cp.reshape(attn_out, (batch_size, seq_len, cfg.n_embd))

            # Output projection
            attn_out = cutile_linear_bias(
                attn_out,
                self.weights[prefix + 'attn.c_proj.weight'],
                self.weights[prefix + 'attn.c_proj.bias'],
                self.weight_transposes.get(prefix + 'attn.c_proj.weight')
            )
            x = x + attn_out

            # MLP
            x_norm = cutile_layer_norm(
                x, self.weights[prefix + 'ln_2.weight'], self.weights[prefix + 'ln_2.bias']
            )
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
        x = cutile_layer_norm(x, self.weights['ln_f.weight'], self.weights['ln_f.bias'])

        # LM head
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
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                top_vals = cp.partition(logits, -k, axis=-1)[:, -k:]
                kth_val = cp.min(top_vals, axis=-1, keepdims=True)
                logits = cp.where(logits < kth_val, float('-inf'), logits)

            # Softmax and sample
            logits_max = cp.max(logits, axis=-1, keepdims=True)
            exp_logits = cp.exp(logits - logits_max)
            probs = exp_logits / cp.sum(exp_logits, axis=-1, keepdims=True)

            batch_size = probs.shape[0]
            idx_next = cp.zeros((batch_size, 1), dtype=cp.int64)
            for b in range(batch_size):
                cumsum_probs = cp.cumsum(probs[b])
                rand_val = cp.random.rand().astype(cp.float32)
                idx_next[b, 0] = cp.searchsorted(cumsum_probs, rand_val)

            idx = cp.concatenate([idx, idx_next], axis=1)

        return idx

    def load_from_huggingface(self, model_name: str = 'gpt2'):
        """
        Load weights from HuggingFace transformers.

        Args:
            model_name: HuggingFace model name (e.g., 'gpt2', 'gpt2-medium')
        """
        from transformers import GPT2LMHeadModel

        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        sd = hf_model.state_dict()

        def to_cupy(tensor):
            return cp.asarray(tensor.detach().cpu().numpy())

        def transpose_contiguous(w):
            w_t = cp.transpose(w)
            if not w_t.flags.c_contiguous:
                w_t = cp.ascontiguousarray(w_t)
            return w_t

        # Embeddings
        self.weights['wte'] = to_cupy(sd['transformer.wte.weight'])
        self.weights['wpe'] = to_cupy(sd['transformer.wpe.weight'])

        # Transformer blocks
        for i in range(self.config.n_layer):
            prefix = f'h.{i}.'
            hf_prefix = f'transformer.h.{i}.'

            # LayerNorm 1
            self.weights[prefix + 'ln_1.weight'] = to_cupy(sd[hf_prefix + 'ln_1.weight'])
            self.weights[prefix + 'ln_1.bias'] = to_cupy(sd[hf_prefix + 'ln_1.bias'])

            # Attention (HuggingFace weights are transposed)
            self.weights[prefix + 'attn.c_attn.weight'] = transpose_contiguous(
                to_cupy(sd[hf_prefix + 'attn.c_attn.weight']))
            self.weights[prefix + 'attn.c_attn.bias'] = to_cupy(sd[hf_prefix + 'attn.c_attn.bias'])
            self.weights[prefix + 'attn.c_proj.weight'] = transpose_contiguous(
                to_cupy(sd[hf_prefix + 'attn.c_proj.weight']))
            self.weights[prefix + 'attn.c_proj.bias'] = to_cupy(sd[hf_prefix + 'attn.c_proj.bias'])

            # LayerNorm 2
            self.weights[prefix + 'ln_2.weight'] = to_cupy(sd[hf_prefix + 'ln_2.weight'])
            self.weights[prefix + 'ln_2.bias'] = to_cupy(sd[hf_prefix + 'ln_2.bias'])

            # MLP (HuggingFace weights are transposed)
            self.weights[prefix + 'mlp.c_fc.weight'] = transpose_contiguous(
                to_cupy(sd[hf_prefix + 'mlp.c_fc.weight']))
            self.weights[prefix + 'mlp.c_fc.bias'] = to_cupy(sd[hf_prefix + 'mlp.c_fc.bias'])
            self.weights[prefix + 'mlp.c_proj.weight'] = transpose_contiguous(
                to_cupy(sd[hf_prefix + 'mlp.c_proj.weight']))
            self.weights[prefix + 'mlp.c_proj.bias'] = to_cupy(sd[hf_prefix + 'mlp.c_proj.bias'])

        # Final LayerNorm
        self.weights['ln_f.weight'] = to_cupy(sd['transformer.ln_f.weight'])
        self.weights['ln_f.bias'] = to_cupy(sd['transformer.ln_f.bias'])

        # LM head (tied with wte)
        self.weights['lm_head.weight'] = self.weights['wte']

        # Recompute transposes
        self._precompute_transposes()

        print(f"✅ Loaded weights from HuggingFace: {model_name}")

    def load_from_mingpt(self, mingpt_model):
        """
        Load weights from a minGPT model (PyTorch).

        Args:
            mingpt_model: A minGPT GPT model instance
        """
        sd = mingpt_model.state_dict()

        def to_cupy(tensor):
            return cp.asarray(tensor.detach().cpu().numpy())

        self.weights['wte'] = to_cupy(sd['transformer.wte.weight'])
        self.weights['wpe'] = to_cupy(sd['transformer.wpe.weight'])

        for i in range(self.config.n_layer):
            prefix = f'h.{i}.'
            sd_prefix = f'transformer.h.{i}.'

            self.weights[prefix + 'ln_1.weight'] = to_cupy(sd[sd_prefix + 'ln_1.weight'])
            self.weights[prefix + 'ln_1.bias'] = to_cupy(sd[sd_prefix + 'ln_1.bias'])
            self.weights[prefix + 'attn.c_attn.weight'] = to_cupy(sd[sd_prefix + 'attn.c_attn.weight'])
            self.weights[prefix + 'attn.c_attn.bias'] = to_cupy(sd[sd_prefix + 'attn.c_attn.bias'])
            self.weights[prefix + 'attn.c_proj.weight'] = to_cupy(sd[sd_prefix + 'attn.c_proj.weight'])
            self.weights[prefix + 'attn.c_proj.bias'] = to_cupy(sd[sd_prefix + 'attn.c_proj.bias'])
            self.weights[prefix + 'ln_2.weight'] = to_cupy(sd[sd_prefix + 'ln_2.weight'])
            self.weights[prefix + 'ln_2.bias'] = to_cupy(sd[sd_prefix + 'ln_2.bias'])
            self.weights[prefix + 'mlp.c_fc.weight'] = to_cupy(sd[sd_prefix + 'mlp.c_fc.weight'])
            self.weights[prefix + 'mlp.c_fc.bias'] = to_cupy(sd[sd_prefix + 'mlp.c_fc.bias'])
            self.weights[prefix + 'mlp.c_proj.weight'] = to_cupy(sd[sd_prefix + 'mlp.c_proj.weight'])
            self.weights[prefix + 'mlp.c_proj.bias'] = to_cupy(sd[sd_prefix + 'mlp.c_proj.bias'])

        self.weights['ln_f.weight'] = to_cupy(sd['transformer.ln_f.weight'])
        self.weights['ln_f.bias'] = to_cupy(sd['transformer.ln_f.bias'])
        self.weights['lm_head.weight'] = to_cupy(sd['lm_head.weight'])

        self._precompute_transposes()
        print("✅ Loaded weights from minGPT")
