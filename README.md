<h1 align="center">cutileGPT</h1>

<p align="center"><strong>A tile-native inference lab for modern decoder LLMs.</strong></p>

<p align="center">
  Graph-aware execution planning above readable
  <a href="https://github.com/NVIDIA/cutile-python">NVIDIA cuTile Python</a> kernels.
</p>

<p align="center">
  <a href="https://github.com/falcons-eyes/cutileGPT/actions/workflows/ci.yml"><img src="https://github.com/falcons-eyes/cutileGPT/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/cutile-gpt/"><img src="https://img.shields.io/pypi/v/cutile-gpt?color=3775A9" alt="PyPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&amp;logoColor=white" alt="Python 3.13+"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-13.1%2B-76B900?logo=nvidia&amp;logoColor=white" alt="CUDA 13.1+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-4C7BD9" alt="Apache-2.0"></a>
</p>

<p align="center">
  <a href="#why-tiles">Why tiles</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#execution-planner">Planner</a>
</p>

---

| **1.80×** | **≈ parity** | **35% fewer launches** |
|:---:|:---:|:---:|
| decode vs PyTorch eager | decode vs `torch.compile` | Qwen3 decode · 350 → 227 |

cutileGPT explores a focused question: **how much framework and kernel-boundary
overhead can tile-native kernels remove without dropping to thread-level CUDA?**

It combines model-graph decisions—fusion, aliasing, prefill/decode specialization,
and backend selection—with cuTile kernels that leave vectorization, register
allocation, shared-memory layout, bank-conflict handling, and hardware scheduling
to the compiler.

> [!IMPORTANT]
> This is an experimental, single-GPU inference project. It is not a training or
> distributed-serving framework.

<p align="center">
  <img src="docs/assets/readme/tile-programming.gif" alt="Tile programming animation: a programmer specifies tiles while the compiler maps them to GPU execution" width="100%">
</p>

## Why tiles

CUDA kernels usually expose how individual threads cooperate. cuTile kernels
describe what a tile computes:

```python
@ct.kernel
def rms_norm_kernel(X, W, Y, eps, N: ConstInt, TILE_N: ConstInt):
    row = ct.bid(0)
    x = ct.load(X, index=(row, 0), shape=(1, TILE_N))
    w = ct.load(W, index=(0,), shape=(TILE_N,))
    rstd = 1 / ct.sqrt(ct.sum(x * x, axis=1) / N + eps)
    ct.store(Y, index=(row, 0), tile=(x * rstd * w).astype(Y.dtype))
```

<sub>Shortened for clarity. See the full [RMSNorm kernel](cutile_gpt/kernels/rmsnorm.py).</sub>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/tile-pipeline-dark.svg">
  <img src="docs/assets/readme/tile-pipeline-light.svg" alt="A Hugging Face checkpoint flows through a graph-aware planner and cuTile kernels to an NVIDIA GPU" width="100%">
</picture>

The current runtime includes:

- RMSNorm, RoPE, MHA/GQA, sliding-window attention, and SwiGLU
- packed QKV and gate/up projections
- fused QK-Norm + RoPE + KV-cache writes
- fused down-projection + residual epilogues
- zero-copy KV-cache views and decode-specialized query tiles
- fixed-shape CUDA Graph capture for static inference
- strict Hugging Face `config.json` + safetensors loading

Fusion is selective. A staged fused MLP is **1.23× faster** than its separate
path, while an earlier mega-kernel duplicated matmul work and was **166× slower**.
The rule is simple: fuse to remove launches or materialization, never to repeat
expensive computation.

## Benchmarks

`Qwen/Qwen3-0.6B`, bf16, batch 1, NVIDIA GB10. Times are milliseconds; lower is
better. Prefill uses a fixed-shape CUDA Graph. Decode advances the real KV cache.

| Phase | Tokens / context | PyTorch eager | `torch.compile` | cutileGPT |
|---|---:|---:|---:|---:|
| Prefill | 128 | 12.84 | **8.10** | 8.57 |
| Prefill | 512 | 24.86 | **15.13** | 15.71 |
| Decode | 128 | 11.72 | 6.75 | **6.50** |
| Decode | 512 | 12.21 | 7.40 | **7.18** |

The honest result: cutileGPT is clearly faster than eager PyTorch, while the
optimized comparison is close—4–6% behind on prefill and 3–4% ahead on these
decode measurements. Treat small gaps as shape- and system-dependent, not as a
universal win.

<details>
<summary><strong>Methodology and reproduction</strong></summary>

CUDA events exclude model loading and compilation. PyTorch uses
`max-autotune-no-cudagraphs`, preserving Inductor fusion and tuning while avoiding
an internal graph capture that conflicts with the advancing KV cache. Nsight
Systems launch counts use one warmed-up decode step.

- [Raw results](docs/assets/qwen3-0.6b-gb10.json)
- [Benchmark](scripts/benchmark_transformer.py)
- [Correctness verifier](scripts/verify_model.py)

```bash
uv run python scripts/verify_model.py Qwen/Qwen3-0.6B
uv run python scripts/benchmark_transformer.py Qwen/Qwen3-0.6B \
  --prefill 128,512 --decode 128,512,2048 --pytorch
uv run python scripts/benchmark_transformer.py Qwen/Qwen3-0.6B \
  --prefill 128,512 --decode 128,512 --pytorch --torch-compile
```

</details>

## Quick start

### Requirements

- Python 3.13+
- NVIDIA driver r580+
- CUDA Toolkit 13.1+
- a GPU supported by the installed `tileiras` compiler

Check the [upstream system requirements](https://github.com/NVIDIA/cutile-python#system-requirements)
for current GPU support.

### Install and run

```bash
pip install "cutile-gpt[hf,torch]"
```

```python
import cupy as cp
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from cutile_gpt.models.transformer import TransformerLM

model_id = "Qwen/Qwen3-0.6B"
path = snapshot_download(model_id)
tokenizer = AutoTokenizer.from_pretrained(path)
model = TransformerLM.from_pretrained(path)

tokens = tokenizer("The future of GPU programming is", return_tensors="np").input_ids
logits = model.forward(cp.asarray(tokens, dtype=cp.int32))
next_id = int(cp.argmax(logits[0, -1]).get())
print(tokenizer.decode([next_id]))
```

For a source checkout:

```bash
git clone --recursive https://github.com/falcons-eyes/cutileGPT.git
cd cutileGPT
uv sync --all-extras
uv run python scripts/verify_model.py Qwen/Qwen3-0.6B
```

## Execution planner

cutileGPT does not position cuTile against `torch.compile`. They solve different
layers of the stack:

```text
model graph   fusion · aliasing · execution phase · backend choice
                         ↓ TileRegion contract
tile compiler tile shape · vectorization · registers · shared memory · scheduling
```

`TileRegion` describes semantic boundaries without prescribing CUDA threads. A
kernel registry measures numerically valid candidates and caches the best tactic
by GPU, dtype, shape, and phase.

```bash
uv run python scripts/autotune_regions.py Qwen/Qwen3-0.6B \
  --prefill 128 --decode 128 --cache .cutile-gpt-tactics.json

uv run python scripts/benchmark_transformer.py Qwen/Qwen3-0.6B \
  --prefill 128 --decode 512 --show-plan \
  --tactic-cache .cutile-gpt-tactics.json
```

## Model support

The loader targets dense, decoder-only, RoPE + RMSNorm + gated-MLP models. It has
been verified against Qwen3, Qwen2.5, Phi-3, Llama 3.2, and SmolLM2 checkpoints.
See the [compatibility matrix](docs/MODELS.md) for exact checkpoints and known
limitations.

Not yet supported: MoE, state-space layers, pre-quantized weights, partial RoPE,
or architectures with per-layer-type head shapes.

## Development

```bash
uvx ruff check .
uv lock --check
uv build
uv run pytest
```

GPU tests require compatible NVIDIA hardware. Contributions and benchmark results
from other GPUs are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE)
