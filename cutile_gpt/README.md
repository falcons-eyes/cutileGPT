# cutile_gpt - Implementation Details

**Pure Tile Programming Philosophy GPU Kernels & Model**

이 디렉토리는 cutileGPT의 핵심 구현을 포함합니다.

## 📁 구조

```
cutile_gpt/
├── model_tile.py              # Pure Tile Philosophy GPT
├── model.py                   # Original CuPy implementation
├── compare.py                 # PyTorch vs cutileGPT comparison
└── kernels/                   # Tile Programming kernels
    ├── layernorm.py          # Declarative LayerNorm
    ├── gelu.py               # 8.3x faster GELU
    ├── linear.py             # Tile-based matmul
    ├── attention.py          # Flash Attention
    └── ...
```

## 🎯 Tile Programming Kernels

### LayerNorm ([layernorm.py](kernels/layernorm.py))

**철학**: Declarative normalization - NO manual synchronization

**특징**:
- Welford's algorithm for numerical stability
- Two-pass approach: statistics → normalize
- Power-of-2 padding for tile constraints
- Automatic thread management

**사용법**:
```python
from cutile_gpt.kernels.layernorm import cutile_layer_norm

x = cp.random.randn(batch, seq, n_embd, dtype=cp.float32)
weight = cp.ones(n_embd, dtype=cp.float32)
bias = cp.zeros(n_embd, dtype=cp.float32)

y = cutile_layer_norm(x, weight, bias)
```

### GELU ([gelu.py](kernels/gelu.py))

**성능**: **8.3x faster** than CuPy! (Verified)

**철학**: Pure element-wise operations, compiler handles parallelization

**특징**:
- GPT-2 style approximation: `0.5 * x * (1 + tanh(...))`
- Automatic vectorization
- No thread management

**사용법**:
```python
from cutile_gpt.kernels.gelu import cutile_gelu

x = cp.random.randn(batch, seq, hidden, dtype=cp.float32)
y = cutile_gelu(x)  # 8.3x faster!
```

**벤치마크** (32 × 512 × 768 tensor):
- Tile kernel: 0.600 ms
- CuPy kernel: 4.978 ms
- **Speedup: 8.3x** (Verified on GB10/Blackwell)

### Linear ([linear.py](kernels/linear.py))

**철학**: Declarative matmul - compiler handles tile operations

**특징**:
- Tile-based matrix multiplication
- Automatic Tensor Core dispatch
- Weight transpose caching (28% speedup)
- 2D swizzle pattern for L2 cache locality
- TMA (Tensor Memory Accelerator) on Hopper/Blackwell

**사용법**:
```python
from cutile_gpt.kernels.linear import cutile_linear_bias

x = cp.random.randn(batch, seq, in_features, dtype=cp.float32)
weight = cp.random.randn(out_features, in_features, dtype=cp.float32) * 0.02
bias = cp.zeros(out_features, dtype=cp.float32)

y = cutile_linear_bias(x, weight, bias)
```

### Attention ([attention.py](kernels/attention.py))

**철학**: Flash Attention - O(N) memory, not O(N²)

**특징**:
- Online softmax algorithm
- Causal masking support
- Multi-head attention
- TMA for async memory transfers
- NO full attention matrix materialization

**사용법**:
```python
from cutile_gpt.kernels.attention import cutile_causal_attention

# Q, K, V: (batch, n_head, seq_len, head_dim)
y = cutile_causal_attention(q, k, v, n_head)
```

## 🎨 Models

### model_tile.py - Pure Tile Philosophy

**완전한 GPT 구현 with ZERO explicit thread management**

**특징**:
- All operations declarative
- Transformer blocks with residual connections
- Text generation support
- minGPT weight loading

**사용법**:
```python
from cutile_gpt.model_tile import create_gpt_nano, CutileGPT, GPTConfig

# Quick start
model = create_gpt_nano()

# Forward pass
tokens = cp.array([[100, 200, 300]], dtype=cp.int32)
logits = model.forward(tokens)

# Generate
generated = model.generate(tokens, max_new_tokens=50)

# Custom config
config = GPTConfig(n_layer=6, n_head=4, n_embd=256)
model = CutileGPT(config)
```

**Available configs**:
- `create_gpt_nano()` - 3 layers, 48 dims, 3 heads
- `create_gpt2('gpt2')` - 12 layers, 768 dims, 12 heads
- `create_gpt2('gpt2-medium')` - 24 layers, 1024 dims, 16 heads

### model.py - Original Implementation

**기존 CuPy 기반 구현 (PyTorch parity 달성)**

**사용법**:
```python
from cutile_gpt.model import CutileGPT, CutileGPTConfig

config = CutileGPTConfig.gpt_tile_medium()
model = CutileGPT(config)

logits, _ = model(idx)
```

## 🔧 최적화 기법

### 1. Weight Transpose Caching
모든 weight transpose를 초기화 시 pre-compute
- **Impact**: 28% average speedup

### 2. Flash Attention
Online softmax로 메모리 효율적
- **Memory**: O(N) instead of O(N²)

### 3. TF32 Tensor Cores
`float32` 입력 자동 TF32 변환
- **Impact**: 8x faster than FP32 CUDA cores

### 4. 2D Swizzle Pattern
L2 cache locality 최적화
- Better cache hit rate

### 5. TMA (Tensor Memory Accelerator)
Hopper/Blackwell 하드웨어 가속
- Async memory transfers

## 📊 성능

### Kernel Level
| Kernel | Tile | CuPy | Speedup |
|--------|------|------|---------|
| GELU (32×512×768) | 0.600 ms | 4.978 ms | **8.3x** (Verified) |
| LayerNorm | Fast | Reference | Competitive |
| Linear | Fast | Reference | Competitive |

### Model Level
| Model | cutileGPT | PyTorch | Result |
|-------|-----------|---------|--------|
| gpt_tile_medium (batch=8, seq=128) | 5.399 ms | 5.174 ms | **Within 4% of PyTorch** |

## 🧪 Testing

```python
# Test individual kernel
python -m cutile_gpt.kernels.gelu

# Test model
python -m cutile_gpt.model_tile

# Compare with PyTorch
python cutile_gpt/compare.py --model nano
```

## 📚 API Reference

### Kernels

**cutile_layer_norm(x, weight, bias, eps=1e-5)**
- Input: `(batch, seq, n_embd)`
- Output: Same shape

**cutile_gelu(x)**
- Input: Any shape
- Output: Same shape
- 8.3x faster than CuPy (Verified)

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

## 🎓 Tile Philosophy 원칙

이 구현의 모든 커널은 다음 원칙을 따릅니다:

1. **Declarative** - WHAT을 명시, HOW는 컴파일러
2. **No thread IDs** - `ct.bid()` only, no `threadIdx`
3. **No synchronization** - No `__syncthreads()`
4. **High-level ops** - `ct.load()`, `ct.sum()`, `ct.mma()`
5. **Compiler-driven** - Automatic optimization

## 🔗 참고

- [demo_tile_gpt.py](../demo_tile_gpt.py) - 완전한 실행 예제
- [TILE_PHILOSOPHY_DEMO.md](../TILE_PHILOSOPHY_DEMO.md) - 철학 문서
- [NVIDIA CUDA Tile Docs](https://docs.nvidia.com/cuda/tile-ir/)

---

**Built with Tile Programming Philosophy** 🚀

*Think in WHAT (operations), not HOW (threads)*
