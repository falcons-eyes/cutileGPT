# cutileGPT Architecture & Vision

## 🎯 Core Philosophy: Tile-Based Thinking

### What is Tile Programming?

**Traditional GPU Programming (CUDA/PTX)**:
```cuda
// Explicit memory management, thread indexing
__global__ void layernorm(float* x, float* y, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;

    // Manual shared memory
    __shared__ float smem[256];

    // Explicit loads
    float val = x[tid];
    smem[threadIdx.x] = val;
    __syncthreads();

    // Manual reduction
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    // ... more manual work
}
```

**Tile Programming Philosophy**:
```mlir
// Declarative: WHAT we want, not HOW
%sum = reduce %x dim=0 identities=[0.0 : f32]
    : !cuda_tile.tile<256xf32> -> !cuda_tile.tile<f32>
(%elem, %acc) {
    %new_acc = addf %elem, %acc
    yield %new_acc
}

// Compiler handles:
// - Thread mapping
// - Shared memory
// - Synchronization
// - Register allocation
```

**Benefits**:
- ✅ **Portable**: Same code, different hardware
- ✅ **Composable**: Tiles are first-class values
- ✅ **Optimizable**: Compiler sees high-level intent
- ✅ **Maintainable**: No manual thread management

---

## 🏗️ cutileGPT Architecture

### Two Implementation Paths

```
┌─────────────────────────────────────────────────────────┐
│                    cutileGPT Project                     │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼                                     ▼
┌──────────────────┐              ┌──────────────────────┐
│   Path 1: API    │              │  Path 2: MLIR/DSL   │
│  "Tile Usage"    │              │ "Tile Philosophy"   │
└──────────────────┘              └──────────────────────┘
        │                                     │
        ▼                                     ▼
```

### Path 1: Python API (cutile_gpt/)

**Status**: ✅ Complete and working

**Approach**: Use CUDA Tile as a library
```python
# cutile_gpt/kernels/layernorm.py
import tile as ct

@ct.kernel
def layernorm(x_ptr, gamma_ptr, beta_ptr, y_ptr, n_embd):
    # Python-based tile API
    x_tile = ct.load(x_ptr, shape=(256,))
    mean = ct.sum(x_tile) / n_embd
    # ... manual tile operations
```

**Characteristics**:
- ✅ Works today
- ✅ Better than raw CUDA
- ⚠️  Still "Tile API usage" not "Tile thinking"
- ⚠️  Python overhead
- ⚠️  Manual optimization needed

**Analogy**: Like using NumPy - better than C loops, but you still write the algorithm

---

### Path 2: MLIR/DSL (cutile_gpt_mlir/)

**Status**: 🚧 Infrastructure built, integration needed

**Approach**: Compile-time optimization via MLIR
```mlir
// cutile_gpt_mlir/kernels/layernorm.mlir
cuda_tile.entry @layernorm_kernel(%x_ptr, %gamma_ptr, %beta_ptr, %y_ptr) {
    // Pure declarative tile operations
    %x = load_ptr_tko weak %x_ptrs : tile<256xf32>

    // Compiler-driven optimization
    %sum = reduce %x dim=0 identities=[0.0 : f32]
        : tile<256xf32> -> tile<f32>
    (%elem, %acc) {
        %new_acc = addf %elem, %acc
        yield %new_acc
    }

    // High-level transformations
    %mean = divf %sum, %n_embd : tile<f32>
    %mean_bc = broadcast %mean : tile<f32> -> tile<256xf32>
    // ...
}
```

**Characteristics**:
- 🎯 True "Tile Philosophy"
- 🎯 Declarative - express WHAT not HOW
- 🎯 Compiler optimization
- 🎯 Hardware abstraction
- 🎯 Provable correctness

**Analogy**: Like writing SQL - you describe WHAT data you want, database engine figures out HOW

---

## 🧩 Component Architecture

### Current Components

```
cutileGPT/
│
├── cutile_gpt/                    # Path 1: Python API
│   ├── model.py                   # ✅ GPT model (working)
│   ├── kernels/
│   │   ├── layernorm.py          # ✅ Tile API usage
│   │   ├── linear_v2.py          # ✅ MatMul with tiles
│   │   └── gelu.py               # ✅ Activation
│   └── inference.py              # ✅ Inference pipeline
│
├── cutile_gpt_mlir/              # Path 2: MLIR/DSL
│   ├── kernels/
│   │   ├── layernorm.mlir        # 🚧 MLIR kernel (needs PTX backend)
│   │   └── test_simple.mlir      # ✅ Validated syntax
│   ├── compiled/                 # 🎯 Target: PTX/CUBIN output
│   ├── kernels.py                # 🚧 Kernel loader (TODO)
│   └── model.py                  # 🚧 MLIR-based model (TODO)
│
├── external/
│   ├── cuda-tile/                # ✅ CUDA Tile compiler
│   └── llvm-project/             # ✅ LLVM/MLIR (built)
│
├── build/
│   ├── llvm/                     # ✅ LLVM/MLIR build
│   └── cuda-tile/                # ✅ CUDA Tile tools
│
└── tools/
    ├── llvm/                     # ✅ Installed MLIR tools
    └── cuda-tile/                # (optional)
```

---

## 🎨 Ideation: Future Components

### 1. **MLIR Kernel Library** 🚧
```
cutile_gpt_mlir/kernels/
├── layernorm.mlir      # Declarative normalization
├── attention.mlir      # Multi-head attention
├── linear.mlir         # Matrix multiplication
├── gelu.mlir          # Activation functions
└── embedding.mlir     # Token + position embedding
```

**Goal**: High-level, composable, hardware-agnostic kernels

---

### 2. **Compilation Pipeline** 🚧
```
MLIR Source → Optimization → PTX/CUBIN → Runtime

layernorm.mlir
    ↓ (cuda-tile-opt)
optimized.mlir
    ↓ (lowering passes - NEEDS INVESTIGATION)
layernorm.ptx
    ↓ (ptxas or JIT)
layernorm.cubin
    ↓ (CuPy loader)
GPU Execution
```

**Status**:
- ✅ MLIR → Bytecode working
- 🚧 Bytecode → PTX unclear (investigating)

---

### 3. **Python Integration Layer** 🚧
```python
# cutile_gpt_mlir/kernels.py
from cuda import cudart
import cupy as cp

class MLIRKernelLoader:
    """Load and execute MLIR-compiled kernels"""

    def __init__(self, kernel_dir="cutile_gpt_mlir/compiled"):
        self.kernels = {}
        self._load_kernels(kernel_dir)

    def load_kernel(self, name, cubin_path):
        # Option A: Load pre-compiled CUBIN
        module = cp.cuda.runtime.moduleLoad(cubin_path)
        self.kernels[name] = module

        # Option B: JIT compile bytecode at runtime
        # (if CUDA Tile provides JIT API)

    def launch(self, name, grid, block, *args):
        kernel = self.kernels[name]
        kernel.launch(grid, block, args)
```

---

### 4. **Hybrid Model** 🎯 (The Vision!)
```python
# cutile_gpt_mlir/model.py
import cupy as cp
from cutile_gpt_mlir.kernels import MLIRKernelLoader

class CutileGPTMLIR:
    """GPT using MLIR-compiled kernels"""

    def __init__(self, config):
        self.config = config
        self.kernels = MLIRKernelLoader()

        # Parameters (same as Path 1)
        self.wte = cp.random.randn(config.vocab_size, config.n_embd)
        self.wpe = cp.random.randn(config.block_size, config.n_embd)
        # ...

    def forward(self, idx):
        B, T = idx.shape

        # Embedding (maybe keep in Python for now)
        tok_emb = self.wte[idx]  # (B, T, C)
        pos_emb = self.wpe[:T]   # (T, C)
        x = tok_emb + pos_emb

        # Transformer blocks - MLIR kernels!
        for layer in range(self.config.n_layer):
            # LayerNorm via MLIR
            x = self.kernels.launch(
                'layernorm',
                grid=(B*T, 1, 1),
                block=(256, 1, 1),
                x, self.ln1_gamma[layer], self.ln1_beta[layer], x_out,
                self.config.n_embd
            )

            # Attention via MLIR
            x = self.kernels.launch('attention', ...)

            # MLP via MLIR
            x = self.kernels.launch('linear', ...)
            x = self.kernels.launch('gelu', ...)
            x = self.kernels.launch('linear', ...)

        # Final layer norm
        x = self.kernels.launch('layernorm', ...)

        # Output projection
        logits = x @ self.wte.T
        return logits
```

**Benefits**:
- 🚀 **Performance**: Compiler-optimized kernels
- 🎯 **Portability**: Same MLIR works on different GPUs
- 🔧 **Maintainability**: High-level kernel descriptions
- 📊 **Tunability**: Compiler can auto-tune for hardware

---

### 5. **Auto-Tuning System** 🎯 (Future)
```python
# cutile_gpt_mlir/autotuner.py
class KernelAutoTuner:
    """Auto-tune MLIR kernels for target hardware"""

    def tune(self, kernel_mlir, search_space):
        # 1. Generate variants with different tile sizes
        variants = self.generate_variants(kernel_mlir, {
            'tile_size': [128, 256, 512],
            'block_size': [128, 256],
            'unroll_factor': [1, 2, 4]
        })

        # 2. Compile all variants
        compiled = [self.compile(v) for v in variants]

        # 3. Benchmark on target GPU
        results = self.benchmark(compiled)

        # 4. Select best
        return results.best()
```

---

## 🎬 The Complete Vision

### Workflow Comparison

**Path 1 (Current - API)**:
```
User writes Python → Tile API → CUDA kernel → GPU
                    ↑
                Manual optimization
```

**Path 2 (Goal - MLIR)**:
```
User writes MLIR → Compiler optimizes → PTX/CUBIN → GPU
                   ↑
              Automatic optimization
              - Tile size selection
              - Memory layout
              - Register allocation
              - Instruction scheduling
```

---

## 📈 Roadmap

### Phase 1: Infrastructure ✅ DONE
- [x] LLVM/MLIR build
- [x] CUDA Tile compiler
- [x] Basic kernel syntax validation

### Phase 2: PTX Backend 🚧 CURRENT
- [ ] Research CUDA Tile backend
- [ ] MLIR → PTX pipeline
- [ ] Test simple kernel end-to-end

### Phase 3: Kernel Library 🎯 NEXT
- [ ] Port layernorm to valid MLIR
- [ ] Implement attention in MLIR
- [ ] Implement linear/matmul in MLIR
- [ ] Implement GELU in MLIR

### Phase 4: Integration 🎯 FUTURE
- [ ] Python kernel loader
- [ ] CutileGPTMLIR model class
- [ ] Inference pipeline
- [ ] Performance benchmarks

### Phase 5: Advanced 🌟 VISION
- [ ] Auto-tuning system
- [ ] Multi-GPU support
- [ ] Mixed precision
- [ ] Kernel fusion optimization

---

## 💡 Key Insights

### Why Two Paths?

**Path 1 (API)**:
- Quick wins, immediate results
- Learning tool
- Fallback if Path 2 is hard

**Path 2 (MLIR)**:
- True innovation
- Research contribution
- Long-term scalability

### The Ideal End State

```python
# User perspective - simple!
from cutile_gpt_mlir import CutileGPTMLIR

model = CutileGPTMLIR.from_pretrained('gpt2')
output = model.generate("Hello", max_length=50)

# Behind the scenes:
# - MLIR kernels compiled for user's GPU
# - Compiler auto-tuned for optimal performance
# - Declarative kernels = maintainable + fast
```

---

## 🎯 Success Criteria

### Technical
- [ ] MLIR kernels compile to working CUBIN
- [ ] Match or exceed Path 1 performance
- [ ] Prove declarative approach works

### Conceptual
- [ ] Demonstrate "Tile Philosophy" advantage
- [ ] Show compiler optimization benefits
- [ ] Create reusable kernel library

### Impact
- [ ] Contribution to CUDA Tile community
- [ ] Educational resource for Tile programming
- [ ] Template for future MLIR GPU projects

---

## 🔬 Research Questions

1. **Compilation**: How does CUDA Tile bytecode get to PTX?
2. **Optimization**: What passes does MLIR provide for GPU?
3. **Runtime**: Do we JIT or AOT compile?
4. **Performance**: Can we match hand-tuned kernels?
5. **Abstraction**: How much can compiler infer?

---

## 📚 Documentation Structure

```
docs/
├── TILE_PHILOSOPHY.md       # Why tile-based thinking
├── API_GUIDE.md            # Path 1 usage
├── MLIR_GUIDE.md           # Path 2 kernel writing
├── COMPILATION.md          # MLIR → GPU pipeline
└── BENCHMARKS.md           # Performance comparison
```

---

## 🚀 Call to Action

**Immediate**: Solve PTX generation (see NEXT_STEPS.md)
**Short-term**: Working end-to-end MLIR kernel
**Long-term**: Full MLIR-based GPT model

**The Journey**: From imperative CUDA to declarative Tile thinking!
