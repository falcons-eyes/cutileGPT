"""Render docs/MODELS.md from the survey plus whatever has been run.

The parser in cutile_gpt.models.hf_config is the ground truth for "supported" -
it is what actually decides whether a checkpoint loads - so this reports its
verdict rather than the looser heuristic the survey script uses for triage.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Models run end to end against transformers, with the result.
VERIFIED = {
    "Qwen/Qwen3-0.6B": "bf16 argmax 100%, generation matches",
    "Qwen/Qwen3-Reranker-0.6B": "bf16 argmax 100%",
    "Qwen/Qwen2.5-0.5B-Instruct": "bf16 argmax 100%, exercises the QKV bias path",
    "HuggingFaceTB/SmolLM2-360M-Instruct": "bf16 argmax 100%, Llama layout",
    "unsloth/Llama-3.2-1B-Instruct":
        "fp32 argmax 100% (max|dlogit| 0.004); bf16 drifts to ~85%, see below",
    "microsoft/Phi-3-mini-4k-instruct":
        "bf16 argmax 100%; fused QKV and gate/up split at load, head_dim 96",
}

META = {
    "meta-models/Muse-Glimmer-30B": ("30B dense", "Meta, Apache-2.0, Aug 2026"),
    "Qwen/Qwen3-32B": ("32B dense", "Alibaba"),
    "Qwen/Qwen3.8-27B": ("27B dense", "Alibaba, Aug 2026"),
    "google/gemma-4-31b-it": ("31B dense", "Google"),
    "mistralai/Mistral-Small-24B-Instruct-2501": ("24B dense", "Mistral"),
    "zai-org/GLM-4-32B-0414": ("32B dense", "Zhipu"),
    "01-ai/Yi-1.5-34B-Chat": ("34B dense", "01.AI"),
    "microsoft/phi-4": ("14B dense", "Microsoft"),
    "Qwen/Qwen3-30B-A3B": ("30B MoE / 3B active", "Alibaba"),
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": ("30B MoE / 3B active", "Alibaba"),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": ("30B hybrid MoE", "NVIDIA"),
    "zai-org/GLM-4.5-Air": ("MoE", "Zhipu"),
    "Qwen/Qwen3-14B": ("14B dense", "Alibaba"),
    "Qwen/Qwen3-0.6B": ("0.6B dense", "Alibaba"),
    "Qwen/Qwen3-Reranker-0.6B": ("0.6B dense", "Alibaba"),
    "Qwen/Qwen2.5-0.5B-Instruct": ("0.5B dense", "Alibaba, Qwen2 layout"),
    "google/gemma-3-27b-it": ("27B dense", "Google, gated"),
    "CohereLabs/c4ai-command-r-v01": ("35B dense", "Cohere, gated"),
    "google/gemma-3-4b-it": ("4B dense", "Google, gated"),
    "meta-llama/Llama-3.2-1B": ("1B dense", "Meta, gated"),
    "unsloth/Llama-3.2-1B-Instruct": ("1B dense", "Llama 3 layout, ungated"),
    "microsoft/Phi-3-mini-4k-instruct": ("3.8B dense", "Microsoft, phi3 layout"),
    "HuggingFaceTB/SmolLM2-360M-Instruct": ("0.36B dense", "Llama layout"),
}

HEAD = """# Open-weight model survey

Which current open-weight checkpoints cutileGPT can load and run.

`config.json` is a few kilobytes and states exactly which primitives a model
needs, so compatibility is decided from it before downloading any weights. The
verdict below comes from `cutile_gpt.models.hf_config.parse_config`, which is
the code that actually refuses or accepts a checkpoint - it is stricter than a
heuristic scan, and deliberately refuses rather than guessing at a field it
does not understand.

```bash
uv run python scripts/survey_architectures.py       # triage a list
uv run python scripts/verify_model.py Qwen/Qwen3-0.6B   # run one against transformers
uv run python scripts/render_models_doc.py          # regenerate this file
```

Legend: ✅ loads and runs · 🔬 additionally verified against transformers ·
❌ refused, with the reason · ⚠️ config not readable.

"""

TAIL = """
Gated repositories answer 401 until the license is accepted on HuggingFace:

```bash
HF_TOKEN=hf_... uv run python scripts/survey_architectures.py
```

## What the refusals mean

**Partial rotary.** Qwen3.8-27B and GLM-4-32B rotate only a fraction of each
head (0.25 and 0.5) and pass the rest through unchanged. The RoPE kernel
rotates the whole head, so these are refused rather than run with the wrong
positions. This is the smallest of the gaps - it is a bound on the rotated
half, not a new kernel.

**Per-layer-type RoPE and shapes.** Gemma 4 gives its global layers a separate
RoPE base, a different `head_dim` (512 against 256), and a different KV head
count from its sliding layers. One architecture record cannot describe that
yet.

**MoE routing.** Qwen3-30B-A3B, Qwen3-Coder-30B, GLM-4.5-Air, and Nemotron 3
keep 128 experts and activate a handful per token. The expert MLPs are ordinary
SwiGLU, so the missing piece is the router and the gather/scatter around it,
not the matmuls.

**State-space layers.** Nemotron 3 Nano also interleaves Mamba-2 with attention
on a fixed pattern and uses squared ReLU rather than a gated MLP - a different
kernel family, not a variation on this one.

**Fused projections are handled, not refused.** Phi packs Q/K/V into one
`qkv_proj` and the gate and up branches into one `gate_up_proj`. The rows are
concatenated in order, so the loader slices them apart once at load and the
forward pass is unchanged. Phi also uses `head_dim` 96, which is not a power of
two - both the RoPE and attention kernels round their tile up and let the extra
lanes load as zero and clip on store.

**Pre-quantized weights.** Anything shipping int4/fp8 needs a dequantizing
loader. `cuda.tile` compiles `float8_e4m3fn`/`e5m2` already, but cupy cannot
import those over DLPack, so they need the direct-torch path.

## Llama and bfloat16

Llama 3 carries enormous activations - the residual stream reaches a magnitude
around 412 from the second layer onward, a known property of the family. One
bfloat16 ulp at that scale is about 2, so the hidden states here and in
transformers differ by ~4 no matter which is "right", and roughly one token in
six flips.

Run in float32 the same checkpoint agrees on argmax at every position with
max|dlogit| 0.004, which is where the arithmetic is actually being checked.
`scripts/verify_model.py --fp32` does this. Qwen and SmolLM2 do not show the
effect and match in bfloat16 directly.

## A note on how these were checked

Agreement is judged on argmax, not on logit values. Both sides run in bfloat16
and accumulate in a different order, so logits drift by a fraction of a unit
against a magnitude around 19. At lower ranks transformers itself rounds two
candidates to the same logit, which makes their ordering a tie-break rather
than a disagreement - the top-1 choice at every position is the meaningful
comparison.
"""


def render() -> str:
    data = json.loads((ROOT / "docs" / "model_survey.json").read_text())

    def row(name: str, info: dict) -> str:
        size, who = META.get(name, ("", ""))
        link = f"[{name}](https://huggingface.co/{name})"
        if "error" in info:
            return f"| {link} | {size} | — | ⚠️ {info['error']} | {who} |"
        shape = (f"{info['layers']}L × {info['hidden']}d, "
                 f"{info['n_head']}/{info['n_kv_head']}")
        if info.get("parser") == "ok":
            mark = "🔬" if name in VERIFIED else "✅"
            note = VERIFIED.get(name, info.get("note", ""))
        else:
            mark, note = "❌", info.get("parser_note", "")
        return f"| {link} | {size} | {shape} | {mark} {note} | {who} |"

    order = [k for k in META if k in data]
    runs = [k for k in order if data[k].get("parser") == "ok"]
    refused = [k for k in order if data[k].get("parser") == "refused"]
    blocked = [k for k in order if "error" in data[k]]

    header = "| Model | Size | Shape | Status | Source |\n|---|---|---|---|---|"
    out = [HEAD]
    out.append(f"## Runs ({len(runs)}, of which {len(VERIFIED)} verified)\n")
    out.append(header)
    out += [row(k, data[k]) for k in runs]
    out.append(f"\n## Refused ({len(refused)})\n")
    out.append(header)
    out += [row(k, data[k]) for k in refused]
    if blocked:
        out.append(f"\n## Config not readable ({len(blocked)})\n")
        out.append(header)
        out += [row(k, data[k]) for k in blocked]
    out.append(TAIL)
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    (ROOT / "docs" / "MODELS.md").write_text(render())
    print("wrote docs/MODELS.md")
