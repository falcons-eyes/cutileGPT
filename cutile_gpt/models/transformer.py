# SPDX-License-Identifier: Apache-2.0
"""A decoder-only transformer driven by a HuggingFace config, not by hardcoding.

`CutileGPT` reproduces GPT-2's graph exactly. This one reads an architecture
description and assembles the same kernels to match whatever the checkpoint
says: RMSNorm, RoPE, grouped-query attention, and a gated MLP, with optional
QK-Norm and per-layer sliding windows.

Weights come in through safetensors and DLPack, so bfloat16 survives - numpy
has no such dtype, and every current checkpoint ships in it.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import cupy as cp

from ..kernels import (
    KVCache,
    cutile_causal_attention,
    cutile_embedding,
    cutile_linear,
    cutile_linear_bias,
    cutile_rms_norm,
    cutile_rope,
    cutile_swiglu_mlp,
    rope_tables,
)
from .hf_config import ModelArchitecture, UnsupportedArchitecture, parse_config


def _to_cupy(tensor) -> cp.ndarray:
    """torch tensor -> cupy, keeping the dtype. See models/gpt.py for why."""
    return cp.from_dlpack(tensor.detach().contiguous().cuda())


class TransformerLM:
    """Inference for a dense GQA/RoPE/RMSNorm/SwiGLU checkpoint."""

    def __init__(self, arch: ModelArchitecture, weights: dict[str, cp.ndarray]):
        self.arch = arch
        self.weights = weights
        self._rope_cache: dict[int, tuple[cp.ndarray, cp.ndarray]] = {}

    # ---------------------------------------------------------------- loading

    @classmethod
    def from_pretrained(cls, path: str | Path, device_check: bool = True
                        ) -> "TransformerLM":
        """Load from a local directory holding config.json and safetensors.

        Use `huggingface_hub.snapshot_download` to fetch one; this deliberately
        does no downloading, so a 60 GB pull is never a side effect of a typo.
        """
        from safetensors import safe_open

        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"{path} is not a directory")

        arch = parse_config(path)

        files = sorted(glob.glob(str(path / "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no .safetensors under {path}")

        weights: dict[str, cp.ndarray] = {}
        for file in files:
            with safe_open(file, framework="pt") as sf:
                for key in sf.keys():
                    weights[key] = _to_cupy(sf.get_tensor(key))

        missing = cls._missing_keys(arch, weights)
        if missing:
            raise UnsupportedArchitecture(
                f"checkpoint is missing {len(missing)} expected tensors, "
                f"first few: {missing[:4]}"
            )

        # Anything in the file that the forward pass does not read would be
        # silently dropped, and a dropped bias produces fluent nonsense rather
        # than an error - which is exactly how Qwen2's QKV biases were missed.
        unused = cls._unconsumed_keys(arch, weights)
        if unused:
            raise UnsupportedArchitecture(
                f"checkpoint carries {len(unused)} tensors this model does not "
                f"consume, so its output would be wrong rather than absent; "
                f"first few: {sorted(unused)[:4]}"
            )
        return cls(arch, weights)

    @staticmethod
    def _consumed_keys(arch: ModelArchitecture, weights: dict) -> set[str]:
        """Every tensor the forward pass actually reads."""
        keys = {"model.embed_tokens.weight", "model.norm.weight"}
        if "lm_head.weight" in weights:
            keys.add("lm_head.weight")
        for i in range(arch.n_layer):
            p = f"model.layers.{i}."
            keys |= {
                p + "input_layernorm.weight",
                p + "post_attention_layernorm.weight",
                p + "self_attn.q_proj.weight",
                p + "self_attn.k_proj.weight",
                p + "self_attn.v_proj.weight",
                p + "self_attn.o_proj.weight",
                p + "mlp.gate_proj.weight",
                p + "mlp.up_proj.weight",
                p + "mlp.down_proj.weight",
            }
            if arch.qk_norm:
                keys |= {p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight"}
            # Qwen2 gives Q, K, and V a bias; Qwen3 and Llama do not.
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                bias = f"{p}self_attn.{proj}.bias"
                if bias in weights:
                    keys.add(bias)
        return keys

    @classmethod
    def _unconsumed_keys(cls, arch: ModelArchitecture, weights: dict) -> set[str]:
        consumed = cls._consumed_keys(arch, weights)
        # rotary_emb.inv_freq is a derived buffer some exports still carry; it
        # is recomputed from rope_theta, so ignoring it changes nothing.
        return {
            k for k in weights
            if k not in consumed and not k.endswith("rotary_emb.inv_freq")
        }

    @staticmethod
    def _missing_keys(arch: ModelArchitecture, weights: dict) -> list[str]:
        expected = ["model.embed_tokens.weight", "model.norm.weight"]
        for i in range(arch.n_layer):
            p = f"model.layers.{i}."
            expected += [
                p + "input_layernorm.weight",
                p + "post_attention_layernorm.weight",
                p + "self_attn.q_proj.weight",
                p + "self_attn.k_proj.weight",
                p + "self_attn.v_proj.weight",
                p + "self_attn.o_proj.weight",
                p + "mlp.gate_proj.weight",
                p + "mlp.up_proj.weight",
                p + "mlp.down_proj.weight",
            ]
            if arch.qk_norm:
                expected += [p + "self_attn.q_norm.weight",
                             p + "self_attn.k_norm.weight"]
        return [k for k in expected if k not in weights]

    # ---------------------------------------------------------------- forward

    def _rope(self, seq_len: int, offset: int):
        key = (seq_len, offset)
        if key not in self._rope_cache:
            self._rope_cache[key] = rope_tables(
                seq_len, self.arch.head_dim, theta=self.arch.rope_theta,
                offset=offset, scaling=self.arch.rope_scaling,
            )
        return self._rope_cache[key]

    def _project(self, x: cp.ndarray, prefix: str) -> cp.ndarray:
        """Linear with the bias applied when the checkpoint carries one."""
        weight = self.weights[prefix + ".weight"]
        bias = self.weights.get(prefix + ".bias")
        if bias is None:
            return cutile_linear(x, weight)
        return cutile_linear_bias(x, weight, bias)

    def _qk_norm(self, x: cp.ndarray, weight: cp.ndarray) -> cp.ndarray:
        """RMSNorm over head_dim, applied per head before RoPE."""
        return cutile_rms_norm(x, weight, eps=self.arch.rms_eps)

    def forward(self, idx: cp.ndarray, cache: KVCache | None = None) -> cp.ndarray:
        """Return logits (batch, seq_len, vocab_size)."""
        arch = self.arch
        batch, seq_len = idx.shape
        past = cache.length if cache is not None else 0

        x = cutile_embedding(idx, self.weights["model.embed_tokens.weight"])
        if arch.embed_scale != 1.0:
            x = x * cp.asarray(arch.embed_scale, dtype=x.dtype)

        cos, sin = self._rope(seq_len, past)

        for i in range(arch.n_layer):
            p = f"model.layers.{i}."

            h = cutile_rms_norm(x, self.weights[p + "input_layernorm.weight"],
                                eps=arch.rms_eps,
                                unit_offset=arch.norm_unit_offset)
            h2d = cp.reshape(h, (-1, arch.hidden_size))

            q = self._project(h2d, p + "self_attn.q_proj")
            k = self._project(h2d, p + "self_attn.k_proj")
            v = self._project(h2d, p + "self_attn.v_proj")

            q = cp.reshape(q, (batch, seq_len, arch.n_head, arch.head_dim))
            k = cp.reshape(k, (batch, seq_len, arch.n_kv_head, arch.head_dim))
            v = cp.reshape(v, (batch, seq_len, arch.n_kv_head, arch.head_dim))

            # QK-Norm runs per head on the last axis, before the transpose.
            if arch.qk_norm:
                q = self._qk_norm(q, self.weights[p + "self_attn.q_norm.weight"])
                k = self._qk_norm(k, self.weights[p + "self_attn.k_norm.weight"])

            q = cp.ascontiguousarray(q.transpose(0, 2, 1, 3))
            k = cp.ascontiguousarray(k.transpose(0, 2, 1, 3))
            v = cp.ascontiguousarray(v.transpose(0, 2, 1, 3))

            if arch.uses_rope(i):
                q = cutile_rope(q, cos, sin)
                k = cutile_rope(k, cos, sin)

            if cache is not None:
                k, v = cache.append(i, k, v)

            attn = cutile_causal_attention(
                q, k, v, arch.n_head, arch.n_kv_head, window=arch.window_for(i))
            attn = cp.ascontiguousarray(attn.transpose(0, 2, 1, 3))
            attn = cp.reshape(attn, (-1, arch.attn_dim))

            x = x + cp.reshape(
                self._project(attn, p + "self_attn.o_proj"),
                (batch, seq_len, arch.hidden_size))

            h = cutile_rms_norm(
                x, self.weights[p + "post_attention_layernorm.weight"],
                eps=arch.rms_eps, unit_offset=arch.norm_unit_offset)
            x = x + cutile_swiglu_mlp(
                h,
                self.weights[p + "mlp.gate_proj.weight"],
                self.weights[p + "mlp.up_proj.weight"],
                self.weights[p + "mlp.down_proj.weight"])

        x = cutile_rms_norm(x, self.weights["model.norm.weight"],
                            eps=arch.rms_eps, unit_offset=arch.norm_unit_offset)

        head = self.weights.get("lm_head.weight")
        if head is None:
            head = self.weights["model.embed_tokens.weight"]  # tied
        return cutile_linear(cp.reshape(x, (-1, arch.hidden_size)), head).reshape(
            batch, seq_len, arch.vocab_size)

    def new_cache(self, batch: int = 1, max_seq_len: int | None = None) -> KVCache:
        arch = self.arch
        return KVCache(
            n_layer=arch.n_layer,
            batch=batch,
            n_kv_head=arch.n_kv_head,
            max_seq_len=max_seq_len or min(arch.max_position, 4096),
            head_dim=arch.head_dim,
            dtype=self.weights["model.embed_tokens.weight"].dtype,
        )

    def __repr__(self) -> str:
        params = sum(w.size for w in self.weights.values())
        return f"TransformerLM({self.arch.describe()}, {params / 1e9:.2f}B params)"


def load_config(path: str | Path) -> ModelArchitecture:
    """Parse just the config, without touching the weights."""
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    return parse_config(json.loads(path.read_text()))
