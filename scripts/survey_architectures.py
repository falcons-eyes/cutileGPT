"""Fetch config.json for a list of models and report what cutileGPT covers.

A model's config.json is a few kilobytes and states exactly which primitives it
needs, so architecture compatibility can be decided without downloading any
weights. Run this before pulling 60 GB of safetensors.

    uv run python scripts/survey_architectures.py
    uv run python scripts/survey_architectures.py --json > docs/model_survey.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

# 30B-class open-weight models, plus a few smaller members of the same families
# that are cheap to verify numerically against.
MODELS = [
    # ~30B dense - the target class
    "meta-models/Muse-Glimmer-30B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3.8-27B",
    "google/gemma-3-27b-it",
    "google/gemma-4-31b-it",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "zai-org/GLM-4-32B-0414",
    "01-ai/Yi-1.5-34B-Chat",
    "CohereLabs/c4ai-command-r-v01",
    "microsoft/phi-4",
    # ~30B sparse (MoE) - same footprint, different routing
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "zai-org/GLM-4.5-Air",
    # smaller models used to verify the loader numerically against transformers
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "unsloth/Llama-3.2-1B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-1B",
]

HF = "https://huggingface.co/{}/resolve/main/config.json"


def fetch(model: str, timeout: float = 20.0) -> tuple[dict | None, str]:
    """Return (config, status). Gated repos answer 401 until a token is set."""
    headers = {"User-Agent": "cutileGPT-survey"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = urllib.request.Request(HF.format(model), headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()), "ok"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return None, "gated"
        if exc.code == 404:
            return None, "not found"
        return None, f"http {exc.code}"
    except (urllib.error.URLError, TimeoutError):
        return None, "unreachable"
    except json.JSONDecodeError:
        return None, "bad config"


def text_config(cfg: dict) -> dict:
    """Multimodal configs nest the language model under text_config."""
    return cfg.get("text_config", cfg)


def classify(cfg: dict) -> dict:
    """Reduce a config to the primitives that decide whether we can run it."""
    t = text_config(cfg)

    n_head = t.get("num_attention_heads")
    n_kv = t.get("num_key_value_heads", n_head)
    # Every family spells these differently.
    n_expert = (t.get("num_experts") or t.get("num_local_experts")
                or t.get("n_routed_experts") or 0)

    act = (t.get("hidden_act") or t.get("hidden_activation")
           or t.get("mlp_hidden_act") or "").lower()
    # SwiGLU/GeGLU multiply a gate branch by an up branch. Squared ReLU
    # (Nemotron) and plain GELU (GPT-2) do not, so they need a different MLP.
    gated = any(g in act for g in ("silu", "swiglu", "geglu")) or act == "gelu_pytorch_tanh"

    # Mamba/attention hybrids interleave state-space layers with attention.
    hybrid = t.get("hybrid_override_pattern") or t.get("layer_types")
    is_hybrid = bool(t.get("hybrid_override_pattern")) or "mamba" in str(
        cfg.get("model_type", "")).lower()

    window = t.get("sliding_window")
    # Some configs carry the field but disable it.
    if window and t.get("use_sliding_window") is False:
        window = None

    # Qwen3 normalizes Q and K per head before RoPE. It is RMSNorm over
    # head_dim, so it composes from kernels we have, but a loader that ignores
    # the q_norm/k_norm tensors would silently produce wrong numbers.
    qk_norm = bool(t.get("use_qk_norm")) or cfg.get("model_type") in ("qwen3",)

    return {
        "model_type": cfg.get("model_type"),
        "qk_norm": qk_norm,
        "architectures": cfg.get("architectures", []),
        "layers": t.get("num_hidden_layers"),
        "hidden": t.get("hidden_size"),
        "n_head": n_head,
        "n_kv_head": n_kv,
        "head_dim": t.get("head_dim"),
        "intermediate": t.get("intermediate_size"),
        "vocab": t.get("vocab_size"),
        "context": t.get("max_position_embeddings"),
        "rope_theta": t.get("rope_theta"),
        "rms_eps": t.get("rms_norm_eps"),
        "activation": act,
        "gated_mlp": gated,
        "gqa": bool(n_head and n_kv and n_kv < n_head),
        "moe": bool(n_expert),
        "n_expert": n_expert,
        "hybrid": is_hybrid,
        "layer_pattern": str(hybrid)[:60] if hybrid else None,
        "sliding_window": window,
        "dtype": t.get("torch_dtype") or cfg.get("torch_dtype"),
        "quantized": bool(cfg.get("quantization_config")),
        "multimodal": "text_config" in cfg,
    }


def verdict(info: dict) -> tuple[str, str]:
    """Can the kernels in this repo express this architecture?"""
    if info["hybrid"]:
        extra = f" + MoE ({info['n_expert']} experts)" if info["moe"] else ""
        return "no", f"Mamba/attention hybrid{extra}; needs state-space layers"
    if info["moe"]:
        return "no", f"MoE routing ({info['n_expert']} experts) not implemented"
    if info["quantized"]:
        return "no", "ships pre-quantized; needs a dequantizing loader"
    if not info["layers"]:
        return "?", "config did not parse"
    if not info["gated_mlp"]:
        return "partial", f"activation {info['activation']!r} is not gated"

    notes = []
    if info["gqa"]:
        notes.append(f"GQA {info['n_head']}/{info['n_kv_head']}")
    if info["qk_norm"]:
        notes.append("QK-Norm")
    if info["sliding_window"]:
        notes.append(f"window {info['sliding_window']}")
    if info["multimodal"]:
        notes.append("text path only")
    return "yes", ", ".join(notes) or "dense MHA"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("models", nargs="*", default=None)
    args = ap.parse_args()

    targets = args.models or MODELS
    results = {}

    for name in targets:
        cfg, status = fetch(name)
        if cfg is None:
            results[name] = {"error": status}
            if not args.json:
                hint = " (set HF_TOKEN)" if status == "gated" else ""
                print(f"  {name:52} {status}{hint}")
            continue
        info = classify(cfg)
        ok, why = verdict(info)
        info["supported"] = ok
        info["note"] = why
        results[name] = info
        if not args.json:
            heads = f"{info['n_head']}/{info['n_kv_head']}" if info["n_head"] else "?"
            print(f"  {name:52} {ok:8} {str(info['layers']):>4}L "
                  f"{str(info['hidden']):>6}d {heads:>8}  {why}")

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
