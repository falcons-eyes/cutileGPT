# Open-weight model survey

Which current open-weight architectures cutileGPT's kernels can express.

A model's `config.json` states exactly which primitives it needs and is a few
kilobytes, so compatibility is decided from it rather than by downloading
weights. Regenerate with:

```bash
uv run python scripts/survey_architectures.py            # table
uv run python scripts/survey_architectures.py --json     # this file's source
```

Legend: ✅ expressible with the kernels here · 🟡 partially · ❌ needs a
primitive that does not exist yet · ⚠️ config not readable.


## Supported (10)

| Model | Size | Shape | Status | Notes |
|---|---|---|---|---|
| [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | 30B dense | 52L × 6656d, 32/2 | ✅ GQA 32/2, window 2048, text path only | Meta, Apache-2.0, Aug 2026 |
| [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 32B dense | 64L × 5120d, 64/8 | ✅ GQA 64/8, QK-Norm | Alibaba |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B dense | 64L × 5120d, 24/4 | ✅ GQA 24/4, text path only | Alibaba, Aug 2026, multimodal |
| [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) | 31B dense | 60L × 5376d, 32/16 | ✅ GQA 32/16, window 1024, text path only | Google |
| [mistralai/Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) | 24B dense | 40L × 5120d, 32/8 | ✅ GQA 32/8 | Mistral |
| [zai-org/GLM-4-32B-0414](https://huggingface.co/zai-org/GLM-4-32B-0414) | 32B dense | 61L × 6144d, 48/2 | ✅ GQA 48/2 | Zhipu |
| [01-ai/Yi-1.5-34B-Chat](https://huggingface.co/01-ai/Yi-1.5-34B-Chat) | 34B dense | 60L × 7168d, 56/8 | ✅ GQA 56/8 | 01.AI |
| [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) | 14B dense | 40L × 5120d, 40/10 | ✅ GQA 40/10 | Microsoft |
| [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | 14B dense | 40L × 5120d, 40/8 | ✅ GQA 40/8, QK-Norm | verification target |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.6B dense | 28L × 1024d, 16/8 | ✅ GQA 16/8, QK-Norm | verification target |

## Not yet supported (4)

| Model | Size | Shape | Status | Notes |
|---|---|---|---|---|
| [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 30B MoE / 3B active | 48L × 2048d, 32/4 | ❌ MoE routing (128 experts) not implemented | Alibaba |
| [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | 30B MoE / 3B active | 48L × 2048d, 32/4 | ❌ MoE routing (128 experts) not implemented | Alibaba |
| [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | 30B hybrid MoE | 52L × 2688d, 32/2 | ❌ Mamba/attention hybrid + MoE (128 experts); needs state-space layers | NVIDIA |
| [zai-org/GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air) | MoE | 46L × 4096d, 96/8 | ❌ MoE routing (128 experts) not implemented | Zhipu |

## Config not readable (4)

| Model | Size | Shape | Status | Notes |
|---|---|---|---|---|
| [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | 27B dense | — | ⚠️ gated | Google, gated |
| [CohereLabs/c4ai-command-r-v01](https://huggingface.co/CohereLabs/c4ai-command-r-v01) | 35B dense | — | ⚠️ gated | Cohere, gated |
| [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | 4B dense | — | ⚠️ gated | Google, gated |
| [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | 1B dense | — | ⚠️ gated | Meta, gated |

Gated repositories return 401 until the license is accepted on HuggingFace and
a token is exported:

```bash
HF_TOKEN=hf_... uv run python scripts/survey_architectures.py
```

## What the gaps are

**MoE routing.** Qwen3-30B-A3B, GLM-4.5-Air, and Nemotron 3 keep 128 experts
and activate a handful per token. The expert MLPs are ordinary SwiGLU - the
missing piece is the router and the gather/scatter around it, not the matmuls.

**State-space layers.** Nemotron 3 Nano interleaves Mamba-2 and attention on a
fixed pattern (`MEMEM*EMEMEM*...`, where `*` is attention) and uses squared
ReLU rather than a gated MLP. That is a different kernel family.

**Pre-quantized checkpoints.** Anything shipping int4/fp8 weights needs a
dequantizing loader. `cuda.tile` compiles `float8_e4m3fn`/`e5m2` already, but
cupy cannot import those over DLPack, so they need the direct-torch path.

