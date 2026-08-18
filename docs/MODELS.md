# Open-weight model survey

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


## Runs (12, of which 6 verified)

| Model | Size | Shape | Status | Source |
|---|---|---|---|---|
| [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | 30B dense | 52L × 6656d, 32/2 | ✅ GQA 32/2, window 2048, text path only | Meta, Apache-2.0, Aug 2026 |
| [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 32B dense | 64L × 5120d, 64/8 | ✅ GQA 64/8, QK-Norm | Alibaba |
| [mistralai/Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) | 24B dense | 40L × 5120d, 32/8 | ✅ GQA 32/8 | Mistral |
| [01-ai/Yi-1.5-34B-Chat](https://huggingface.co/01-ai/Yi-1.5-34B-Chat) | 34B dense | 60L × 7168d, 56/8 | ✅ GQA 56/8 | 01.AI |
| [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) | 14B dense | 40L × 5120d, 40/10 | ✅ GQA 40/10; fused QKV, fused gate/up | Microsoft |
| [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | 14B dense | 40L × 5120d, 40/8 | ✅ GQA 40/8, QK-Norm | Alibaba |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.6B dense | 28L × 1024d, 16/8 | 🔬 bf16 argmax 100%, generation matches | Alibaba |
| [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 0.6B dense | 28L × 1024d, 16/8 | 🔬 bf16 argmax 100% | Alibaba |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 0.5B dense | 24L × 896d, 14/2 | 🔬 bf16 argmax 100%, exercises the QKV bias path | Alibaba, Qwen2 layout |
| [unsloth/Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) | 1B dense | 16L × 2048d, 32/8 | 🔬 fp32 argmax 100% (max|dlogit| 0.004); bf16 drifts to ~85%, see below | Llama 3 layout, ungated |
| [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | 3.8B dense | 32L × 3072d, 32/32 | 🔬 bf16 argmax 100%; fused QKV and gate/up split at load, head_dim 96 | Microsoft, phi3 layout |
| [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 0.36B dense | 32L × 960d, 15/5 | 🔬 bf16 argmax 100%, Llama layout | Llama layout |

## Refused (7)

| Model | Size | Shape | Status | Source |
|---|---|---|---|---|
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B dense | 64L × 5120d, 24/4 | ❌ partial_rotary_factor 0.25 rotates only part of each head; not implemented | Alibaba, Aug 2026 |
| [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) | 31B dense | 60L × 5376d, 32/16 | ❌ per-layer-type RoPE parameters (['full_attention', 'sliding_attention']) are not implemented; this model applies a different base per layer type | Google |
| [zai-org/GLM-4-32B-0414](https://huggingface.co/zai-org/GLM-4-32B-0414) | 32B dense | 61L × 6144d, 48/2 | ❌ partial_rotary_factor 0.5 rotates only part of each head; not implemented | Zhipu |
| [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 30B MoE / 3B active | 48L × 2048d, 32/4 | ❌ MoE routing (128 experts) is not implemented | Alibaba |
| [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | 30B MoE / 3B active | 48L × 2048d, 32/4 | ❌ MoE routing (128 experts) is not implemented | Alibaba |
| [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | 30B hybrid MoE | 52L × 2688d, 32/2 | ❌ MoE routing (128 experts) is not implemented | NVIDIA |
| [zai-org/GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air) | MoE | 46L × 4096d, 96/8 | ❌ MoE routing (128 experts) is not implemented | Zhipu |

## Config not readable (4)

| Model | Size | Shape | Status | Source |
|---|---|---|---|---|
| [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | 27B dense | — | ⚠️ gated | Google, gated |
| [CohereLabs/c4ai-command-r-v01](https://huggingface.co/CohereLabs/c4ai-command-r-v01) | 35B dense | — | ⚠️ gated | Cohere, gated |
| [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | 4B dense | — | ⚠️ gated | Google, gated |
| [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | 1B dense | — | ⚠️ gated | Meta, gated |

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

