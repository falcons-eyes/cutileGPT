# cutile_gpt - Implementation Details

**Pure Tile Programming Philosophy GPU Kernels & Model**

This directory contains the core implementation of cutileGPT.

## 📁 Structure

```
cutile_gpt/
├── kernels/                   # Tile Programming kernels
│   ├── layernorm.py          # Declarative LayerNorm
│   ├── gelu.py               # 8.3x faster GELU
│   ├── linear.py             # Tile-based matmul
│   ├── attention.py          # Flash Attention
│   ├── embedding.py          # Token + position embedding
│   └── fused_mlp.py          # Fused fc -> GELU -> proj
├── models/                    # Model assembly
│   ├── gpt.py                # CutileGPT
│   └── config.py             # GPTConfig and its presets
├── api/                       # High-level Tile API
│   ├── tile_op.py            # TileOp fluent builder
│   ├── config.py             # TileConfig, TensorSpec, Layout, DType
│   └── profiler.py           # DataAnalyzer, DataProfile
├── utils/                     # Benchmarking and HF weight loading
└── examples/                  # Annotated single-kernel tutorials
```

## 🎯 Tile Programming Kernels

### LayerNorm ([layernorm.py](kernels/layernorm.py))

**Philosophy**: Declarative normalization - NO manual synchronization

**Features**:
- Welford's algorithm for numerical stability
- Two-pass approach: statistics → normalize
- Power-of-2 padding for tile constraints
- Automatic thread management

**Usage**:
```python
from cutile_gpt.kernels.layernorm import cutile_layer_norm

x = cp.random.randn(batch, seq, n_embd, dtype=cp.float32)
weight = cp.ones(n_embd, dtype=cp.float32)
bias = cp.zeros(n_embd, dtype=cp.float32)

y = cutile_layer_norm(x, weight, bias)
```

### GELU ([gelu.py](kernels/gelu.py))

**Performance**: **8.3x faster** than CuPy

**Philosophy**: Pure element-wise operations, compiler handles parallelization

**Features**:
- GPT-2 style approximation: `0.5 * x * (1 + tanh(...))`
- Automatic vectorization
- No thread management

**Usage**:
```python
from cutile_gpt.kernels.gelu import cutile_gelu

x = cp.random.randn(batch, seq, hidden, dtype=cp.float32)
y = cutile_gelu(x)
```

**Benchmark** (32 × 512 × 768 tensor):
- Tile kernel: 0.600 ms
- CuPy kernel: 4.978 ms
- **Speedup: 8.3x**

### Linear ([linear.py](kernels/linear.py))

**Philosophy**: Declarative matmul - compiler handles tile operations

**Features**:
- Tile-based matrix multiplication
- Automatic Tensor Core dispatch
- Weight transpose caching
- 2D swizzle pattern for L2 cache locality
- TMA (Tensor Memory Accelerator) on Hopper/Blackwell

**Usage**:
```python
from cutile_gpt.kernels.linear import cutile_linear_bias

x = cp.random.randn(batch, seq, in_features, dtype=cp.float32)
weight = cp.random.randn(out_features, in_features, dtype=cp.float32) * 0.02
bias = cp.zeros(out_features, dtype=cp.float32)

y = cutile_linear_bias(x, weight, bias)
```

### Attention ([attention.py](kernels/attention.py))

**Philosophy**: Flash Attention - O(N) memory, not O(N²)

**Features**:
- Online softmax algorithm
- Causal masking support
- Multi-head attention
- TMA for async memory transfers
- NO full attention matrix materialization

**Usage**:
```python
from cutile_gpt.kernels.attention import cutile_causal_attention

# Q, K, V: (batch, n_head, seq_len, head_dim)
y = cutile_causal_attention(q, k, v, n_head)
```

## 🎨 Models

### models/gpt.py - CutileGPT

**Complete GPT implementation with ZERO explicit thread management**

**Features**:
- All operations declarative
- Transformer blocks with residual connections
- Text generation support
- HuggingFace and minGPT weight loading

**Usage**:
```python
import cupy as cp
from cutile_gpt import CutileGPT, GPTConfig

# Quick start from a preset
model = CutileGPT(GPTConfig.gpt_nano())

# Forward pass - returns (logits, loss) where loss is always None
tokens = cp.array([[100, 200, 300]], dtype=cp.int32)
logits, _ = model.forward(tokens)

# Generate
generated = model.generate(tokens, max_new_tokens=50)

# Custom config
model = CutileGPT(GPTConfig(n_layer=6, n_head=4, n_embd=256))
```

### models/config.py - GPTConfig presets

`GPTConfig` is a dataclass (`vocab_size`, `block_size`, `n_layer`, `n_head`,
`n_embd`) with classmethod presets:

| Preset | Layers | Heads | Embedding | Context |
|--------|--------|-------|-----------|---------|
| `GPTConfig.gpt_nano()` | 3 | 3 | 48 | 128 |
| `GPTConfig.gpt_micro()` | 4 | 4 | 128 | 256 |
| `GPTConfig.gpt_mini()` | 6 | 6 | 192 | 256 |
| `GPTConfig.gpt2()` | 12 | 12 | 768 | 1024 |
| `GPTConfig.gpt2_medium()` | 24 | 16 | 1024 | 1024 |
| `GPTConfig.gpt2_large()` | 36 | 20 | 1280 | 1024 |
| `GPTConfig.gpt2_xl()` | 48 | 25 | 1600 | 1024 |
| `GPTConfig.gpt_tile_small()` | 4 | 4 | 64 | 128 |
| `GPTConfig.gpt_tile_medium()` | 6 | 4 | 128 | 256 |
| `GPTConfig.gpt_tile_large()` | 8 | 8 | 256 | 512 |

`GPTConfig.from_name("gpt2")` resolves a preset by name.

Loading pretrained weights requires the `hf` extra:

```python
model = CutileGPT(GPTConfig.gpt2())
model.load_from_huggingface("gpt2")   # pip install cutile-gpt[hf]
```

## 🔧 Optimization Techniques

### 1. Weight Transpose Caching
Pre-compute all weight transposes during initialization
- Reduces runtime overhead

### 2. Flash Attention
Online softmax for memory efficiency
- **Memory**: O(N) instead of O(N²)

### 3. TF32 Tensor Cores
Automatic TF32 conversion for `float32` inputs
- 8x faster than FP32 CUDA cores

### 4. 2D Swizzle Pattern
L2 cache locality optimization
- Better cache hit rate

### 5. TMA (Tensor Memory Accelerator)
Hopper/Blackwell hardware acceleration
- Async memory transfers

## 📊 Performance

### Kernel Level
| Kernel | Tile | CuPy | Speedup |
|--------|------|------|---------|
| GELU (32×512×768) | 0.600 ms | 4.978 ms | **8.3x** |
| LayerNorm | Fast | Reference | Competitive |
| Linear | Fast | Reference | Competitive |

### Model Level
| Model | cutileGPT | PyTorch | Result |
|-------|-----------|---------|--------|
| gpt_tile_medium (batch=8, seq=128) | 5.399 ms | 5.174 ms | **Competitive** |

## 🧪 Testing

```python
# Test individual kernel
python -m cutile_gpt.kernels.gelu

# Run the end-to-end demo
python demo_tile_gpt.py

# Compare with PyTorch minGPT
python scripts/compare_mingpt.py
```

## 📚 API Reference

### Kernels

**cutile_layer_norm(x, weight, bias, eps=1e-5)**
- Input: `(batch, seq, n_embd)`
- Output: Same shape

**cutile_gelu(x)**
- Input: Any shape
- Output: Same shape
- 8.3x faster than CuPy

**cutile_linear_bias(x, weight, bias, weight_t=None)**
- Input: `(..., in_features)`
- Weight: `(out_features, in_features)`
- Output: `(..., out_features)`

**cutile_causal_attention(q, k, v, n_head)**
- Input: `(batch, n_head, seq_len, head_dim)`
- Output: Same shape

### Model

**CutileGPT(config)**
- `forward(idx)` - Forward pass
- `generate(idx, max_new_tokens, temperature, top_k)` - Generate text
- `load_from_mingpt(mingpt_model)` - Load PyTorch weights

## 🎓 Tile Philosophy Principles

All kernels in this implementation follow these principles:

1. **Declarative** - Specify WHAT, compiler handles HOW
2. **No thread IDs** - `ct.bid()` only, no `threadIdx`
3. **No synchronization** - No `__syncthreads()`
4. **High-level ops** - `ct.load()`, `ct.sum()`, `ct.mma()`
5. **Compiler-driven** - Automatic optimization

## 🔗 References

- [demo_tile_gpt.py](../demo_tile_gpt.py) - Complete executable demo
- [Project README](../README.md) - Architecture, benchmarks, and quick start
- [NVIDIA CUDA Tile Docs](https://docs.nvidia.com/cuda/tile-ir/)

---

**Built with Tile Programming Philosophy** 🚀

*Think in WHAT (operations), not HOW (threads)*
