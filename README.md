# cutileGPT

> **100% PyTorch-Free GPT implementation using NVIDIA cutile + CuPy**

A high-performance GPT implementation leveraging NVIDIA's cutile framework for tile-based GPU programming. Through careful optimization, cutileGPT **matches PyTorch performance** while maintaining a **dramatically smaller dependency footprint** (~10MB vs ~2GB).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-13.0%2B-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.13%2B-3776ab.svg)](https://www.python.org/)

---

## 🚀 Performance Results

**cutileGPT now matches PyTorch performance!**

### AS-IS vs TO-BE Comparison

Tested on NVIDIA GB10 (Blackwell, SM_121):

| Implementation | Latency (ms) | Throughput (tok/s) | Result |
|----------------|--------------|-------------------|---------|
| **PyTorch minGPT (AS-IS)** | 5.209 | 196.2K | Baseline |
| **cutileGPT (TO-BE)** | 5.175 | 197.1K | ✅ **1.01x faster** |

*Workload: tile-medium (6 layers, 128 dims), batch=8, seq=128, vocab=50257*

### Detailed Performance by Model Size

| Model | Layers | Dims | Latency (ms) | Throughput (tok/s) |
|-------|--------|------|--------------|-------------------|
| **gpt_nano** | 3 | 48 | 1.03 | 248K |
| **gpt_tile_small** | 4 | 64 | 0.96 | 268K |
| **gpt_micro** | 4 | 128 | 1.14 | 225K |
| **gpt_tile_medium** | 6 | 128 | 1.34 | 191K |
| **gpt_mini** | 6 | 192 | 2.67 | 96K |
| **gpt_tile_large** | 8 | 256 | 2.63 | 97K |

*Workload: batch=4, seq=64, 100 iterations*

---

## 📊 Interactive Performance Dashboard

View detailed profiling results with interactive charts:

**[📈 Open Performance Dashboard](profiling_results/performance_dashboard.html)**

The dashboard includes:
- Forward pass latency comparison across all model configs
- Throughput analysis (tokens/sec)
- Latency percentile distribution (min, median, p95, p99, max)
- Model architecture breakdown
- Efficiency metrics (throughput per layer)

Raw profiling data: [`profiling_results/profiling_data.json`](profiling_results/profiling_data.json)

---

## 🎯 Why cutileGPT?

### Performance
✅ **Matches PyTorch** - 1.01x faster on realistic workloads
✅ **Lightweight** - ~10MB vs PyTorch's ~2GB
✅ **Zero Overhead** - No autograd, no dispatch layer
✅ **Optimized Kernels** - TF32, TMA, Flash Attention

### Deployment
✅ **Edge-Ready** - 200x smaller footprint
✅ **Docker-Friendly** - Faster builds, lower storage
✅ **Serverless** - Lambda-compatible size
✅ **Pure CuPy** - No PyTorch dependency for inference

### Development
✅ **Educational** - Learn GPU programming with clean kernels
✅ **Transparent** - Every kernel call is explicit
✅ **Customizable** - Easy to modify and optimize
✅ **Modern** - Uses latest NVIDIA features (cutile, TMA, Hopper/Blackwell)

---

## 🔧 Key Optimizations

### 1. **Weight Transpose Caching** 🚀

Pre-compute and cache all weight transposes during initialization:

```python
def _precompute_transposes(self):
    """Cache weight transposes to avoid runtime overhead."""
    for key, weight in self.weights.items():
        if 'weight' in key and weight.ndim == 2:
            weight_t = cp.transpose(weight)
            self.weight_transposes[key] = cp.ascontiguousarray(weight_t)
```

**Impact**: 28% average speedup, up to 33% on large models

### 2. **Removed Fused MLP** ❌

The fused MLP kernel degraded performance by **14x** on realistic workloads. Reverted to simple, proven approach:

```python
# Before (Fused MLP): 14x slower
mlp_out = cutile_fused_mlp(x, w_fc, b_fc, w_proj, b_proj)

# After (Separate ops): Fast and correct
hidden = cutile_linear_bias(x, w_fc, b_fc, w_fc_t)
hidden = cutile_gelu(hidden)
mlp_out = cutile_linear_bias(hidden, w_proj, b_proj, w_proj_t)
```

**Lesson**: cuBLAS-optimized matmul is incredibly hard to beat. Simple solutions often win.

### 3. **Flash Attention**

Online softmax algorithm eliminates full attention matrix materialization:

```python
# Single-pass attention with O(N) memory instead of O(N²)
for j in range(Tc):
    qk = ct.mma(q, k_tile, qk)  # Compute QK^T tile
    p = ct.exp2(qk * scale - m_ij)  # Online softmax
    acc = ct.mma(p, v_tile, acc)  # Accumulate weighted values
```

### 4. **TF32 Tensor Cores**

Automatic TF32 acceleration for `float32` inputs on Ampere+:

```python
dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
a = ct.load(A, ...).astype(dtype)  # Auto-converts to TF32
```

**Impact**: 8x faster than FP32 CUDA cores with minimal accuracy loss

### 5. **2D Swizzle Pattern**

Optimizes L2 cache locality for matrix multiplication:

```python
def swizzle_2d(M, N, tm, tn):
    """Reorder blocks for better cache hits."""
    bid = ct.bid(0)
    group_id = bid // (GROUP_SIZE_M * num_bid_n)
    bid_m = first_bid_m + (bid % group_size_m)
    return bid_m, bid_n
```

### 6. **TMA (Tensor Memory Accelerator)**

Hardware-accelerated async memory transfers on Hopper/Blackwell:

```python
a = ct.load(A, index=(bid_m, k), shape=(tm, tk),
            padding_mode=zero_pad, latency=4, allow_tma=True)
```

---

## 📦 Architecture

cutileGPT is built on a clean 3-layer architecture with **zero PyTorch dependencies**:

```
┌─────────────────────────────────────┐
│   Application Layer                 │
│   (Pure CuPy model implementation)  │
├─────────────────────────────────────┤
│   Array Management (CuPy)           │
│   (NumPy-compatible GPU arrays)     │
├─────────────────────────────────────┤
│   GPU Kernels (NVIDIA cutile)       │
│   (Custom CUDA tile-based kernels)  │
├─────────────────────────────────────┤
│   Hardware Layer                    │
│   (Hopper SM_100, Blackwell SM_120+)│
└─────────────────────────────────────┘
```

---

## 📁 Project Structure

```
cutileGPT/
├── cutile_gpt/              # 100% CuPy implementation (No PyTorch!)
│   ├── kernels/             # Custom CUDA kernels using cutile
│   │   ├── attention.py     # Flash Attention with online softmax
│   │   ├── linear.py        # MatMul with 2D swizzle + TMA + weight caching
│   │   ├── layernorm.py     # LayerNorm with Welford algorithm
│   │   ├── gelu.py          # GELU activation (GPT-2 approximation)
│   │   └── embedding.py     # Embedding lookup (gather op)
│   ├── model.py             # CutileGPT model class (CuPy-based)
│   └── compare.py           # Benchmark script (PyTorch vs CuPy)
├── profiling_results/       # Performance profiling data
│   ├── performance_dashboard.html  # Interactive dashboard
│   ├── profiling_data.json         # Raw benchmark data
│   └── cutile_nsys.nsys-rep        # NVIDIA Nsight Systems profile
├── scripts/                 # Profiling automation
│   ├── run_nsys_profile.sh  # Nsight Systems profiling
│   └── run_ncu_profile.sh   # Nsight Compute profiling
├── external/                # Git submodules
│   ├── cutile-python/       # NVIDIA cutile framework
│   └── minGPT/              # Karpathy's minGPT (reference only)
├── visualize_performance.py # Generate performance dashboard
├── profile_performance.py   # Detailed profiling script
├── test_text_generation.py # Text generation with GPT-2 tokenizer
├── pyproject.toml           # Project configuration
├── OPTIMIZATION_SUMMARY.md  # Detailed optimization journey
└── README.md
```

---

## 🚀 Setup

### Prerequisites

- **Python 3.13+**
- **CUDA 13.0+**
- **NVIDIA GPU** with compute capability 10.0+ (Hopper) or 12.0+ (Blackwell)
  - Tested on: GB10 (Blackwell, SM_121)

### Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/falcons-eyes/cutileGPT.git
cd cutileGPT

# Or if already cloned, initialize submodules
git submodule update --init --recursive

# Install dependencies with uv
uv sync
```

**Note**: PyTorch is **NOT** required for cutileGPT inference. It's only used in `compare.py` for benchmarking against the reference implementation.

---

## 💻 Usage

### Quick Start (CuPy-only inference)

```python
import cupy as cp
from cutile_gpt.model import CutileGPT, CutileGPTConfig

# Create a tile-optimized model (power-of-2 dimensions)
config = CutileGPTConfig.gpt_tile_small()
model = CutileGPT(config)

# Create input token indices (CuPy array)
batch_size, seq_len = 1, 16
idx = cp.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=cp.int64)

# Forward pass
logits, _ = model(idx)
print(f"Logits shape: {logits.shape}")  # (1, 16, 50257)

# Autoregressive generation
generated = model.generate(
    idx[:, :5],           # Start with 5 tokens
    max_new_tokens=20,    # Generate 20 more
    temperature=0.8,
    top_k=40
)
print(f"Generated: {cp.asnumpy(generated[0]).tolist()}")
```

### Text Generation with GPT-2 Tokenizer

```python
import cupy as cp
from transformers import GPT2Tokenizer
from cutile_gpt.model import CutileGPT, CutileGPTConfig

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Create model
config = CutileGPTConfig.gpt2()
model = CutileGPT(config)

# Tokenize prompt
prompt = "The future of artificial intelligence is"
encoded = tokenizer.encode(prompt, return_tensors='pt')
idx = cp.array([encoded[0].tolist()], dtype=cp.int64)

# Generate
generated = model.generate(idx, max_new_tokens=50, temperature=0.9, top_k=50)

# Decode
output_text = tokenizer.decode(cp.asnumpy(generated[0]).tolist())
print(output_text)
```

### Load Weights from minGPT (PyTorch)

```python
import torch
from mingpt.model import GPT as minGPT
from cutile_gpt.model import CutileGPT, CutileGPTConfig

# Load pretrained minGPT model
mingpt_model = minGPT.from_pretrained('gpt2')
mingpt_model.eval()

# Create cutileGPT with matching config
config = CutileGPTConfig.gpt2()
cutile_model = CutileGPT(config)

# Transfer weights (PyTorch → CuPy via NumPy)
cutile_model.load_from_mingpt(mingpt_model)

# Now use CuPy for inference
import cupy as cp
idx = cp.array([[1, 2, 3, 4, 5]], dtype=cp.int64)
logits, _ = cutile_model(idx)
```

### Run Correctness Test

Verify numerical accuracy against PyTorch minGPT:

```bash
uv run python cutile_gpt/compare.py --model nano
```

### Run Performance Benchmarks

```bash
# Compare AS-IS (PyTorch) vs TO-BE (cutileGPT)
uv run python cutile_gpt/compare.py --benchmark --model tile-medium --batch-size 8 --seq-len 128 --vocab-size 50257

# Generate interactive performance dashboard
uv run python visualize_performance.py --run-profiling

# Profile with NVIDIA Nsight Systems
bash scripts/run_nsys_profile.sh
```

---

## 📋 Available Model Configurations

| Config | Layers | Heads | Embedding Dim | Block Size | Description |
|--------|--------|-------|---------------|------------|-------------|
| `gpt_nano` | 3 | 3 | 48 | 128 | Minimal test config |
| `gpt_micro` | 4 | 4 | 128 | 256 | Small test config |
| `gpt_mini` | 6 | 6 | 192 | 256 | Medium test config |
| `gpt2` | 12 | 12 | 768 | 1024 | GPT-2 (124M params) |
| **`gpt_tile_small`** | 4 | 4 | **64** | 128 | **Power-of-2 optimized** |
| **`gpt_tile_medium`** | 6 | 4 | **128** | 256 | **Power-of-2 optimized** |
| **`gpt_tile_large`** | 8 | 8 | **256** | 512 | **Power-of-2 optimized** |

**Note**: Tile-optimized configs (`gpt_tile_*`) use power-of-2 embedding dimensions, eliminating tile padding overhead and maximizing GPU utilization.

---

## 📊 Profiling & Visualization

### Generate Performance Dashboard

```bash
# Run comprehensive profiling
uv run python visualize_performance.py --run-profiling

# Open the interactive dashboard
open profiling_results/performance_dashboard.html
```

### Custom Profiling

```bash
# Custom batch size and sequence length
uv run python visualize_performance.py --run-profiling --batch-size 16 --seq-len 256

# Profile with NVIDIA Nsight Systems
bash scripts/run_nsys_profile.sh

# View nsys results
nsys-ui profiling_results/cutile_nsys.nsys-rep
```

### Profiling Data Format

The `profiling_data.json` contains structured benchmark results:

```json
{
  "metadata": {
    "timestamp": "2026-01-26T14:56:21.406610",
    "gpu_name": "NVIDIA GB10",
    "compute_capability": "12.1",
    "cuda_version": "13000"
  },
  "benchmarks": {
    "gpt_tile_medium": {
      "config": {"n_layer": 6, "n_head": 4, "n_embd": 128},
      "timing": {
        "mean": 1.34, "std": 0.10, "min": 1.19, "max": 1.84,
        "median": 1.34, "p95": 1.51, "p99": 1.82
      },
      "throughput_tokens_per_sec": 191079
    }
  }
}
```

---

## 🔬 Technical Deep Dive

### Tile-Based Programming with cutile

cutile (cuda.tile) is NVIDIA's framework for writing high-performance GPU kernels using tile-based abstractions:

```python
import cuda.tile as ct

@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1), occupancy=4)
def matmul_kernel(A, B, C, TM: ct.Constant[int], TN: ct.Constant[int], TK: ct.Constant[int]):
    # Get tile indices with 2D swizzle
    bid_m, bid_n = swizzle_2d(M, N, TM, TN)

    # Accumulator in fp32 for precision
    acc = ct.full((TM, TN), 0, dtype=ct.float32)

    # K-loop with TMA and TF32
    for k in range(num_tiles_k):
        # Load tiles from global memory (async with TMA)
        a = ct.load(A, index=(bid_m, k), shape=(TM, TK),
                    padding_mode=ct.PaddingMode.ZERO, latency=4, allow_tma=True)
        b = ct.load(B, index=(k, bid_n), shape=(TK, TN),
                    padding_mode=ct.PaddingMode.ZERO, latency=4, allow_tma=True)

        # Tile-based matrix multiply (uses Tensor Cores)
        dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
        acc = ct.mma(a.astype(dtype), b.astype(dtype), acc)

    # Store result back to global memory
    ct.store(C, index=(bid_m, bid_n), tile=acc.astype(C.dtype))
```

**Key Concepts**:
- **Tiles**: Fixed-size blocks of data (must be power-of-2)
- **TMA**: Hardware async copy from global → shared memory
- **`ct.mma()`**: Automatic Tensor Core dispatch
- **Memory Hierarchy**: Explicit control over register/shared/global

### Flash Attention Implementation

Our Flash Attention kernel implements the online softmax algorithm:

1. **Tiled QK^T**: Compute attention scores in tiles
2. **Online Softmax**: Update running max and sum without materializing full matrix
3. **Weighted Sum**: Compute output as running weighted sum

**Memory Savings**:
- Standard attention: O(N²) memory for full attention matrix
- Flash attention: O(N) memory (only tiles in registers/shared)

**Reference**: [Flash Attention Paper](https://arxiv.org/abs/2205.14135)

---

## 📈 Optimization Journey

See [`OPTIMIZATION_SUMMARY.md`](OPTIMIZATION_SUMMARY.md) for detailed breakdown of:
- Initial bottleneck analysis
- Fused MLP removal (eliminated 14x slowdown)
- Weight transpose caching (28% speedup)
- Performance progression to PyTorch parity

---

## 🛣️ Roadmap

### Completed ✅
- [x] **PyTorch parity** - Achieved 1.01x speedup
- [x] **Weight transpose caching** - 28% average improvement
- [x] **Interactive dashboard** - Plotly-based visualization
- [x] **NVIDIA profiling** - nsys/ncu integration
- [x] **Text generation** - GPT-2 tokenizer support

### Future Work
- [ ] **FP16/BF16 support** - Mixed precision for 2-3x speedup
- [ ] **KV cache** for generation (reduce recomputation)
- [ ] **Multi-GPU support** via CuPy's NCCL integration
- [ ] **INT8 quantization** kernels for Hopper Tensor Cores
- [ ] **Triton backend** (alternative to cutile for portability)

---

## 📚 API Reference

### `CutileGPTConfig`

Configuration dataclass for model architecture.

**Factory Methods**:
- `gpt_nano()` - 3 layers, 48 dims, 3 heads
- `gpt_micro()` - 4 layers, 128 dims, 4 heads
- `gpt_mini()` - 6 layers, 192 dims, 6 heads
- `gpt2()` - 12 layers, 768 dims, 12 heads (124M params)
- `gpt_tile_small()` - 4 layers, **64 dims**, 4 heads (power-of-2)
- `gpt_tile_medium()` - 6 layers, **128 dims**, 4 heads (power-of-2)
- `gpt_tile_large()` - 8 layers, **256 dims**, 8 heads (power-of-2)

### `CutileGPT`

Main model class for inference.

**Constructor**:
```python
CutileGPT(config: CutileGPTConfig, device: str = 'cuda')
```

**Methods**:
- `forward(idx: cp.ndarray) -> Tuple[cp.ndarray, None]`
  - **Input**: Token indices `(batch, seq_len)` as CuPy array
  - **Output**: Logits `(batch, seq_len, vocab_size)`

- `generate(idx: cp.ndarray, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> cp.ndarray`
  - Autoregressive generation
  - Returns extended sequence `(batch, seq_len + max_new_tokens)`

- `load_from_mingpt(mingpt_model)`
  - Load pretrained weights from PyTorch minGPT model
  - Converts tensors to CuPy arrays via NumPy bridge

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Format**: Use `ruff format` for code formatting
2. **Linting**: Run `ruff check` before committing
3. **Testing**: Ensure `compare.py` passes correctness tests
4. **Kernels**: Maintain cutile best practices (power-of-2 tiles, explicit shapes)

---

## 📖 Citation

If you use cutileGPT in your research or project, please cite:

```bibtex
@software{cutilegpt2026,
  title={cutileGPT: PyTorch-Free GPT with NVIDIA cutile},
  author={Falcon Eyes},
  year={2026},
  url={https://github.com/falcons-eyes/cutileGPT}
}
```

---

## 📄 License

Apache-2.0

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **NVIDIA cutile**: Tile-based GPU programming framework ([cutile-python](https://github.com/NVIDIA/cutile-python))
- **Karpathy's minGPT**: Reference PyTorch implementation ([minGPT](https://github.com/karpathy/minGPT))
- **CuPy**: NumPy-compatible GPU arrays ([CuPy](https://cupy.dev/))
- **Flash Attention**: Online softmax algorithm ([Paper](https://arxiv.org/abs/2205.14135))

---

**Built with 💚 using NVIDIA cutile and CuPy**
