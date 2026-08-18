# SPDX-License-Identifier: Apache-2.0
"""
GPT Model Implementation using Tile Programming

A unified GPT implementation using cutile kernels for GPU acceleration.
Supports loading weights from HuggingFace and minGPT.
"""

import math
from typing import Dict, Optional, Tuple

import cupy as cp

from ..kernels import (
    cutile_causal_attention,
    cutile_embedding,
    cutile_gelu,
    cutile_layer_norm,
    cutile_linear,
    cutile_linear_bias,
)
from ..kernels.kv_cache import KVCache
from .config import GPTConfig


def _torch_to_cupy(tensor):
    """Move a torch weight onto the GPU as a CuPy array, preserving its dtype.

    Goes through DLPack rather than `cp.asarray(t.cpu().numpy())`. numpy has no
    bfloat16, so the old route raised "Got unsupported ScalarType BFloat16" on
    every current open-weight checkpoint - GPT-2 only worked because it ships
    fp32. DLPack hands the buffer over directly, which also skips the round trip
    through host memory.
    """
    import cupy as cp

    return cp.from_dlpack(tensor.detach().contiguous().cuda())


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

    def forward(self, idx: cp.ndarray, cache: "KVCache | None" = None
                ) -> Tuple[cp.ndarray, None]:
        """
        Forward pass using cutile kernels.

        Args:
            idx: Token indices (batch, seq_len). With a cache, only the tokens
                not already in it - one per step while decoding.
            cache: Optional KVCache. When given, K and V for this call are
                appended to it and attention runs against the full history, so
                the prefix is not recomputed. Positions are taken from the
                cache length, which is what keeps the position embedding right
                for a single-token call.

        Returns:
            Tuple of (logits, None)
        """
        batch_size, seq_len = idx.shape
        cfg = self.config
        past_len = cache.length if cache is not None else 0
        total_len = past_len + seq_len
        assert total_len <= cfg.block_size, (
            f"Sequence length {total_len} > block_size {cfg.block_size}"
        )

        # Embeddings
        tok_emb = cutile_embedding(idx, self.weights['wte'])
        pos = cp.arange(past_len, total_len, dtype=cp.int64)
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

            # Attention. With a cache, K and V cover the whole history while
            # Q holds only the new tokens, and the kernel derives the query
            # offset from the length difference.
            if cache is not None:
                k, v = cache.append(i, k, v)
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
        top_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> cp.ndarray:
        """
        Autoregressive generation.

        Args:
            idx: Initial token indices (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens
            use_cache: Keep K and V across steps instead of recomputing the
                prefix every token. Without it the cost grows with the square
                of the sequence. Pass False only to compare against that.

        Returns:
            Extended token sequence (batch, seq_len + max_new_tokens)
        """
        cache = None
        if use_cache:
            head_dim = self.config.n_embd // self.config.n_head
            cache = KVCache(
                n_layer=self.config.n_layer,
                batch=idx.shape[0],
                n_kv_head=self.config.n_head,
                max_seq_len=self.config.block_size,
                head_dim=head_dim,
                dtype=self.weights['wte'].dtype,
            )

        for step in range(max_new_tokens):
            if cache is None:
                idx_cond = (idx if idx.shape[1] <= self.config.block_size
                            else idx[:, -self.config.block_size:])
            elif step == 0:
                idx_cond = idx          # prefill the whole prompt at once
            else:
                idx_cond = idx[:, -1:]  # then one token per step

            logits, _ = self.forward(idx_cond, cache=cache)
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

        Raises:
            ImportError: If transformers is not installed
        """
        try:
            from transformers import GPT2LMHeadModel
        except ImportError:
            raise ImportError(
                "load_from_huggingface requires 'transformers' package. "
                "Install with: pip install cutile-gpt[hf]"
            )

        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        sd = hf_model.state_dict()

        to_cupy = _torch_to_cupy

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

        to_cupy = _torch_to_cupy

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
