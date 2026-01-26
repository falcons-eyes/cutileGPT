# cutileGPT

GPT implementation using NVIDIA cutile for optimized CUDA kernels.

This project compares a standard PyTorch minGPT implementation (AS-IS) with a cutile-optimized version (TO-BE) to demonstrate performance benefits of custom CUDA kernels.

## Project Structure

```
cutileGPT/
├── cutile_gpt/           # cutile-optimized GPT implementation
│   ├── kernels/          # Custom CUDA kernels using cutile
│   │   ├── attention.py  # Flash Attention style fused kernel
│   │   ├── linear.py     # Optimized MatMul with 2D swizzle, TMA
│   │   ├── layernorm.py  # LayerNorm with Welford algorithm
│   │   ├── gelu.py       # GELU activation
│   │   ├── embedding.py  # Embedding lookup
│   │   └── fused_mlp.py  # Fused MLP kernel
│   ├── model.py          # CutileGPT model class
│   └── compare.py        # Benchmark script
├── external/             # External dependencies (git submodules)
│   ├── cutile-python/    # NVIDIA cutile framework
│   └── minGPT/           # Karpathy's minGPT (reference)
├── pyproject.toml        # Project configuration
└── README.md
```

## Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/falcons-eyes/cutileGPT.git
cd cutileGPT

# Or if already cloned, init submodules
git submodule update --init --recursive

# Install dependencies with uv
uv sync
```

## Requirements

- Python 3.13+
- CUDA 13.0+
- PyTorch 2.10+
- NVIDIA GPU (tested on GB10/Blackwell)

## Usage

### Run Correctness Test

```bash
uv run python cutile_gpt/compare.py
```

### Run Performance Benchmark

```bash
uv run python cutile_gpt/compare.py --benchmark --model tile-small
uv run python cutile_gpt/compare.py --benchmark --model tile-medium
uv run python cutile_gpt/compare.py --benchmark --model tile-large
```

### Available Model Configurations

| Config | Layers | Heads | Embedding Dim | Description |
|--------|--------|-------|---------------|-------------|
| nano | 3 | 3 | 48 | Minimal test config |
| micro | 4 | 4 | 128 | Small test config |
| mini | 6 | 6 | 192 | Medium test config |
| tile-small | 4 | 4 | 64 | Power-of-2 optimized (small) |
| tile-medium | 6 | 4 | 128 | Power-of-2 optimized (medium) |
| tile-large | 8 | 8 | 256 | Power-of-2 optimized (large) |

## Performance Results

Benchmark results on NVIDIA GB10 (SM_121):

| Config | PyTorch (AS-IS) | cutile (TO-BE) | Speedup |
|--------|-----------------|----------------|---------|
| tile-small | 0.812 ms | 0.591 ms | **1.37x** |
| tile-medium | 1.093 ms | 0.836 ms | **1.31x** |
| tile-large | 1.436 ms | 1.093 ms | **1.31x** |

## Key Optimizations

1. **ByTarget Tuning** - Architecture-specific `num_ctas` and `occupancy` settings
2. **TMA (Tensor Memory Accelerator)** - Hardware-accelerated tile movement
3. **Flash Attention** - Fused QK^T → softmax → AV with online softmax
4. **2D Swizzle** - L2 cache locality optimization for MatMul
5. **TF32 Tensor Cores** - Automatic TF32 conversion for float32 inputs
6. **Welford Algorithm** - Single-pass mean+variance for LayerNorm
7. **Epilogue Fusion** - Fused bias addition in MatMul

## License

Apache-2.0
