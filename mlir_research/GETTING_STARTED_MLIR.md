# Getting Started with MLIR Implementation

## 🎯 Purpose

This guide will help you build and run the **TRUE Tile Philosophy** implementation of cutileGPT using MLIR.

**Why MLIR?**
- ✅ Declarative programming (describe WHAT, not HOW)
- ✅ Compiler-driven optimization
- ✅ Hardware abstraction
- ✅ True tile-based thinking

Compare with Python API (cutile_gpt/):
- Python: "Tile API usage" - manually optimized, PTX-style
- MLIR: "Tile Philosophy" - compiler-optimized, declarative

---

## 📋 Prerequisites

### Required
- **CMake 3.20+**: `cmake --version`
- **C++17 compiler**: GCC 9+ or Clang 10+
- **Python 3.6+**: `python3 --version`
- **CUDA Toolkit 13.1+**: `nvcc --version`
- **Git**: For submodules

### Recommended
- **Ninja**: Faster builds (`sudo apt install ninja-build`)
- **ccache**: Faster recompilation (`sudo apt install ccache`)
- **20GB+ disk space**: For LLVM build
- **16GB+ RAM**: For parallel compilation

### Time Estimate
- **First build**: 1.5-2 hours (LLVM compilation)
- **Subsequent builds**: 5-10 minutes

---

## 🚀 Quick Start

### Step 1: Run Setup Script

```bash
# Make executable
chmod +x setup_cuda_tile.sh

# Run (will take 1.5-2 hours)
./setup_cuda_tile.sh
```

This script will:
1. Clone LLVM at the correct commit
2. Build LLVM/MLIR with optimal settings
3. Build CUDA Tile IR
4. Install tools to `tools/` directory
5. Create environment setup script

**Coffee break recommended ☕** - This takes a while!

### Step 2: Set Up Environment

```bash
# Source environment (do this in every new terminal)
source setup_env.sh

# Verify installation
cuda-tile-translate --help
mlir-opt --help
```

### Step 3: Build MLIR Kernels

```bash
# Configure CMake
cmake -G Ninja -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUTILEGPT_BUILD_MLIR=ON

# Build kernels
cmake --build build

# Install (copies kernels to cutile_gpt_mlir/compiled/)
cmake --install build --prefix .
```

### Step 4: Test MLIR Kernels

```bash
# Test kernel loader
uv run python -m cutile_gpt_mlir.kernels

# Should see:
# ✓ Loaded MLIR kernel: layernorm_kernel from layernorm.cubin
# ✓ Loaded 1 MLIR kernels
```

---

## 📁 Project Structure

After setup, you'll have:

```
cutileGPT/
├── setup_cuda_tile.sh          # Setup script
├── setup_env.sh                # Environment script (generated)
├── CMakeLists.txt              # Build configuration
│
├── external/
│   ├── cuda-tile/              # CUDA Tile submodule
│   └── llvm-project/           # LLVM source (cloned by script)
│
├── tools/
│   ├── llvm/                   # LLVM installation
│   └── cuda-tile/              # CUDA Tile installation
│
├── build/
│   ├── llvm/                   # LLVM build
│   ├── cuda-tile/              # CUDA Tile build
│   └── kernels/                # Compiled kernels (.tilebc, .cubin)
│
├── cutile_gpt/                 # Python API implementation
│   └── kernels/                # "Tile API usage" style
│
└── cutile_gpt_mlir/            # MLIR implementation
    ├── kernels/
    │   ├── layernorm.mlir      # MLIR source
    │   └── CMakeLists.txt
    ├── compiled/               # Installed kernels
    │   ├── layernorm.tilebc    # Bytecode (JIT)
    │   └── layernorm.cubin     # Cubin (AoT)
    ├── kernels.py              # Kernel loader
    └── model.py                # MLIR-based model (TODO)
```

---

## 🔧 Development Workflow

### Writing a New MLIR Kernel

1. **Create MLIR file**: `cutile_gpt_mlir/kernels/mykernel.mlir`

```mlir
cuda_tile.module @mykernel_module {
    cuda_tile.entry @mykernel(
        %input_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %output_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>
    ) {
        // Declarative tile operations
        %token = make_token : !cuda_tile.token

        %data, %token2 = load_ptr_tko weak %input_ptr token=%token
            : !cuda_tile.tile<!cuda_tile.ptr<f32>>
            -> !cuda_tile.tile<128xf32>, !cuda_tile.token

        // ... operations ...

        %token3 = store_ptr_tko weak %output_ptr, %result token=%token2
            : !cuda_tile.tile<!cuda_tile.ptr<f32>>, !cuda_tile.tile<128xf32>
            -> !cuda_tile.token

        return
    }
}
```

2. **Add to CMakeLists.txt**: `cutile_gpt_mlir/kernels/CMakeLists.txt`

```cmake
add_mlir_kernel(mykernel_kernel
    MLIR_FILE ${CMAKE_CURRENT_SOURCE_DIR}/mykernel.mlir
    GPU_ARCH ${GPU_ARCH}
    AOT
)
```

3. **Rebuild**:

```bash
cmake --build build
cmake --install build --prefix .
```

4. **Load in Python**:

```python
from cutile_gpt_mlir import MLIRKernelLoader

loader = MLIRKernelLoader()
loader.load_kernel("mykernel", "mykernel.cubin", "mykernel")

kernel = loader.get_kernel("mykernel")
kernel.launch(grid=(1,1,1), block=(1,1,1), args=[input_ptr, output_ptr])
```

---

## 🐛 Troubleshooting

### Problem: "cuda-tile-translate not found"

**Solution**: Source the environment script

```bash
source setup_env.sh
```

### Problem: "LLVM build takes too long"

**Solutions**:
1. Use more cores: Set `NCORES` in script
2. Use pre-built LLVM (if available)
3. Reduce build type to `MinSizeRel`

### Problem: "MLIR syntax errors"

**Check**:
1. MLIR syntax is strict - check types carefully
2. All operations must have explicit types
3. Token-based ordering for memory ops

**Validate**:
```bash
mlir-opt cutile_gpt_mlir/kernels/layernorm.mlir --verify-diagnostics
```

### Problem: "Kernel launch fails"

**Debug**:
1. Check grid/block dimensions
2. Verify tile sizes match kernel expectations
3. Check pointer arguments are on device

```python
# Enable CUDA error checking
import cupy as cp
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
cp.cuda.Device().synchronize()  # Force synchronous execution for debugging
```

---

## 📚 Learning Resources

### CUDA Tile IR
- [Official Specification](https://docs.nvidia.com/cuda/tile-ir/13.1/)
- [GitHub Repository](https://github.com/NVIDIA/cuda-tile)
- [Example Kernels](external/cuda-tile/test/Dialect/CudaTile/)

### MLIR
- [MLIR Documentation](https://mlir.llvm.org/)
- [MLIR Tutorials](https://mlir.llvm.org/getting_started/)

### Project Documentation
- [CUDA_TILE_PHILOSOPHY_ANALYSIS.md](CUDA_TILE_PHILOSOPHY_ANALYSIS.md) - Why MLIR?
- [CUDA_TILE_MLIR_INTEGRATION_ANALYSIS.md](CUDA_TILE_MLIR_INTEGRATION_ANALYSIS.md) - How to integrate

---

## 🎯 Next Steps

### Phase 1: Foundation (✓ You are here)
- [x] Setup LLVM/MLIR
- [x] Setup CUDA Tile
- [x] Build LayerNorm kernel

### Phase 2: Core Kernels (TODO)
- [ ] Linear (MatMul) kernel
- [ ] GELU activation
- [ ] Attention kernel

### Phase 3: Model Integration (TODO)
- [ ] `CutileGPTMLIR` class
- [ ] Forward pass with MLIR kernels
- [ ] Performance comparison

### Phase 4: Optimization (TODO)
- [ ] Tuned tile sizes
- [ ] Multi-stage pipeline
- [ ] Fusion opportunities

---

## 💪 You're Ready!

Once setup is complete, you'll have:

✅ **Working MLIR development environment**
✅ **Compiled LayerNorm kernel**
✅ **Python interface to load kernels**
✅ **Build system for new kernels**

**This is the foundation for TRUE Tile Philosophy implementation!**

Run the setup script and grab some coffee ☕

```bash
./setup_cuda_tile.sh
```

See you on the other side! 🚀
