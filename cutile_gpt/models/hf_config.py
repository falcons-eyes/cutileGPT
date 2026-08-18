# SPDX-License-Identifier: Apache-2.0
"""Read a HuggingFace `config.json` into the shape the kernels need.

Families spell the same field several ways, and a loader that quietly falls
back to a default produces wrong numbers rather than an error - so anything
that would change the arithmetic is either read or refused here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Architectures this can express: dense, gated MLP, RoPE, RMSNorm, GQA.
GATED_ACTIVATIONS = {"silu", "swish", "silu_and_mul", "geglu", "gelu_pytorch_tanh"}


class UnsupportedArchitecture(Exception):
    """Raised for a config whose arithmetic these kernels cannot reproduce."""


@dataclass
class ModelArchitecture:
    """Everything about a checkpoint that changes the forward pass."""

    model_type: str
    n_layer: int
    hidden_size: int
    n_head: int
    n_kv_head: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    max_position: int
    rope_theta: float
    rms_eps: float
    tie_word_embeddings: bool
    qk_norm: bool
    sliding_window: int
    # Gemma scales embeddings by sqrt(hidden) and offsets norm weights by 1.
    embed_scale: float = 1.0
    norm_unit_offset: bool = False
    layer_windows: list[int] = field(default_factory=list)
    rope_layers: list[bool] = field(default_factory=list)

    @property
    def attn_dim(self) -> int:
        """n_head * head_dim, which is not always hidden_size."""
        return self.n_head * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_head * self.head_dim

    def window_for(self, layer_idx: int) -> int:
        """0 means a global layer."""
        if self.layer_windows:
            return self.layer_windows[layer_idx]
        return self.sliding_window

    def uses_rope(self, layer_idx: int) -> bool:
        """Glimmer leaves its global layers unrotated."""
        if self.rope_layers:
            return self.rope_layers[layer_idx]
        return True

    def describe(self) -> str:
        gqa = f"{self.n_head}/{self.n_kv_head}"
        bits = [f"{self.n_layer}L", f"{self.hidden_size}d", f"GQA {gqa}",
                f"head_dim {self.head_dim}", f"theta {self.rope_theta:g}"]
        if self.qk_norm:
            bits.append("QK-Norm")
        if self.sliding_window:
            bits.append(f"window {self.sliding_window}")
        return f"{self.model_type}: " + ", ".join(bits)


def _text_config(cfg: dict) -> dict:
    """Multimodal checkpoints nest the language model under text_config."""
    return cfg.get("text_config", cfg)


def _require(cfg: dict, *names: str):
    for name in names:
        if cfg.get(name) is not None:
            return cfg[name]
    raise UnsupportedArchitecture(
        f"config is missing all of {names}; refusing to guess"
    )


def parse_config(config: dict | str | Path) -> ModelArchitecture:
    """Parse a config dict, a path to config.json, or a directory holding one."""
    if isinstance(config, (str, Path)):
        path = Path(config)
        if path.is_dir():
            path = path / "config.json"
        config = json.loads(path.read_text())

    model_type = config.get("model_type", "")
    t = _text_config(config)

    if config.get("quantization_config"):
        raise UnsupportedArchitecture(
            "checkpoint ships pre-quantized weights; needs a dequantizing loader"
        )
    n_expert = (t.get("num_experts") or t.get("num_local_experts")
                or t.get("n_routed_experts") or 0)
    if n_expert:
        raise UnsupportedArchitecture(
            f"MoE routing ({n_expert} experts) is not implemented"
        )
    if t.get("hybrid_override_pattern") or "mamba" in model_type.lower():
        raise UnsupportedArchitecture(
            "Mamba/attention hybrid; state-space layers are not implemented"
        )

    act = (t.get("hidden_act") or t.get("hidden_activation")
           or t.get("mlp_hidden_act") or "").lower()
    if act not in GATED_ACTIVATIONS:
        raise UnsupportedArchitecture(
            f"activation {act!r} is not a gated MLP; only SwiGLU/GeGLU is implemented"
        )

    n_layer = _require(t, "num_hidden_layers")
    hidden = _require(t, "hidden_size")
    n_head = _require(t, "num_attention_heads")
    n_kv_head = t.get("num_key_value_heads", n_head)
    head_dim = t.get("head_dim") or hidden // n_head

    if n_head % n_kv_head != 0:
        raise UnsupportedArchitecture(
            f"n_head {n_head} is not divisible by n_kv_head {n_kv_head}"
        )

    # RoPE base hides in three places depending on the family, and defaulting
    # to 10000 when the checkpoint says 500000 produces a model that runs and
    # is wrong - so it is read, not assumed.
    rope_params = t.get("rope_parameters") or {}
    if rope_params and "rope_theta" not in rope_params:
        # Gemma 4 nests a separate block per layer type, with different bases
        # and a partial rotary factor. That is more than one theta can express.
        raise UnsupportedArchitecture(
            f"per-layer-type RoPE parameters ({sorted(rope_params)}) are not "
            "implemented; this model applies a different base per layer type"
        )
    rope_theta = t.get("rope_theta") or rope_params.get("rope_theta")

    layer_theta = t.get("layer_rope_theta") or []
    if layer_theta:
        bases = {float(x) for x in layer_theta if x}
        if len(bases) > 1:
            raise UnsupportedArchitecture(
                f"per-layer RoPE bases {sorted(bases)} are not implemented"
            )
        if bases:
            rope_theta = bases.pop()

    if rope_theta is None:
        raise UnsupportedArchitecture(
            "config states no RoPE base; refusing to assume one"
        )

    if t.get("partial_rotary_factor", 1.0) != 1.0:
        raise UnsupportedArchitecture(
            f"partial_rotary_factor {t['partial_rotary_factor']} rotates only "
            "part of each head; not implemented"
        )

    # Gemma 4 gives global layers their own head_dim and KV head count.
    if t.get("global_head_dim") and t["global_head_dim"] != t.get("head_dim"):
        raise UnsupportedArchitecture(
            f"global layers use head_dim {t['global_head_dim']} against "
            f"{t.get('head_dim')} for local ones; per-layer shapes are not implemented"
        )
    if (t.get("num_global_key_value_heads")
            and t["num_global_key_value_heads"] != t.get("num_key_value_heads")):
        raise UnsupportedArchitecture(
            "global and local layers use different KV head counts; not implemented"
        )

    window = t.get("sliding_window") or 0
    if window and t.get("use_sliding_window") is False:
        window = 0

    # Gemma alternates local and global layers; the pattern is per-layer.
    layer_windows: list[int] = []
    layer_types = t.get("layer_types")
    if layer_types and window:
        layer_windows = [
            window if str(kind).startswith("sliding") else 0 for kind in layer_types
        ]

    # A zero in layer_rope_theta means that layer gets no rotation at all -
    # Glimmer leaves its global layers unrotated so they stay position-agnostic
    # over long range.
    rope_layers = [bool(x) for x in layer_theta] if layer_theta else []

    is_gemma = model_type.startswith("gemma")

    return ModelArchitecture(
        model_type=model_type,
        n_layer=n_layer,
        hidden_size=hidden,
        n_head=n_head,
        n_kv_head=n_kv_head,
        head_dim=head_dim,
        intermediate_size=_require(t, "intermediate_size"),
        vocab_size=_require(t, "vocab_size"),
        max_position=t.get("max_position_embeddings", 4096),
        rope_theta=float(rope_theta),
        rms_eps=float(t.get("rms_norm_eps", t.get("layer_norm_eps", 1e-6))),
        tie_word_embeddings=bool(t.get("tie_word_embeddings", False)),
        qk_norm=bool(t.get("use_qk_norm")) or model_type == "qwen3",
        sliding_window=window,
        embed_scale=float(hidden) ** 0.5 if is_gemma else 1.0,
        norm_unit_offset=is_gemma,
        layer_windows=layer_windows,
        rope_layers=rope_layers,
    )
