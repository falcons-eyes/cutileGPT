# Next Steps for cutileGPT MLIR Implementation

## ✅ Completed

1. **LLVM/MLIR Build** (1.5 hours)
   - 409 MLIR libraries built successfully
   - Tools installed: mlir-opt, mlir-translate, etc.
   - Location: `tools/llvm/`

2. **CUDA Tile Build** (10 minutes)
   - cuda-tile-opt and cuda-tile-translate built
   - Location: `build/cuda-tile/bin/`

3. **Test Kernel Compilation**
   - Created `test_simple.mlir` → compiled to bytecode
   - Verified CUDA Tile pipeline works

## 🔧 Known Issues

### 1. CUDA Tile Installation Failed
**Error**: FileCheck not found during `ninja install`
**Workaround**: Use binaries directly from `build/cuda-tile/bin/`
**Impact**: Minimal - tools work fine from build directory

### 2. layernorm.mlir Needs Updates
**Issues**:
- Uses `cast` instead of `itof` for type conversion
- May use unsupported operations like `cuda_tile.func`

**Required Changes**:
- Replace `cast` with `itof ... signed : tile<i32> -> tile<f32>`
- Verify all operations exist in CUDA Tile dialect
- Reference: `external/cuda-tile/test/Dialect/CudaTile/`

### 3. Bytecode → PTX/CUBIN Missing
**Current**: MLIR → CUDA Tile Bytecode (.tilebc)
**Needed**: Bytecode → PTX/CUBIN for GPU execution

**Options**:
a) CUDA Tile may have a separate backend tool
b) May need NVVM/NVPTX lowering passes
c) Could use CUDA Driver API to JIT compile bytecode

## 📋 Immediate Next Steps

### Step 1: Fix layernorm.mlir
```bash
# 1. Update operations
vim cutile_gpt_mlir/kernels/layernorm.mlir
# - Replace `cast` with `itof`
# - Remove `cuda_tile.func` (use only `cuda_tile.entry`)
# - Simplify to match test examples

# 2. Validate
cuda-tile-opt cutile_gpt_mlir/kernels/layernorm.mlir

# 3. Compile to bytecode
cuda-tile-translate cutile_gpt_mlir/kernels/layernorm.mlir \
    --mlir-to-cudatilebc \
    -o cutile_gpt_mlir/compiled/layernorm.tilebc
```

### Step 2: Research PTX/CUBIN Generation
```bash
# Check CUDA Tile documentation
ls external/cuda-tile/docs/

# Look for backend tools
find build/cuda-tile/bin -type f -executable

# Check for PTX lowering passes
cuda-tile-opt --help | grep -i "ptx\|nvvm\|cubin"

# Investigate LLVM backend
find external/cuda-tile -name "*PTX*" -o -name "*NVVM*"
```

### Step 3: Investigate CUDA Tile Runtime
CUDA Tile bytecode may need a runtime loader:
```bash
# Check for runtime libraries
find build/cuda-tile/lib -name "*.so" -o -name "*.a"

# Look for Python bindings (if built)
find build/cuda-tile -name "*.py"

# Check examples
find external/cuda-tile -name "example*" -o -name "test*" -type f
```

### Step 4: Alternative Approach - Direct PTX
If CUDA Tile doesn't provide PTX backend:

```bash
# Use MLIR's NVVM dialect directly
mlir-opt cutile_gpt_mlir/kernels/layernorm.mlir \
    --convert-cuda-tile-to-nvvm \  # (if exists)
    --gpu-to-nvvm \
    --gpu-to-cubin

# Or use LLVM backend
mlir-translate layernorm.mlir \
    --mlir-to-llvmir \
    | llc -march=nvptx64 -o layernorm.ptx

# Then compile PTX to CUBIN
ptxas layernorm.ptx -o layernorm.cubin
```

## 🎯 Ultimate Goal

### Working Pipeline
```
layernorm.mlir
    ↓ (cuda-tile-opt - optimization)
layernorm_opt.mlir
    ↓ (cuda-tile-translate or custom pass)
layernorm.ptx
    ↓ (ptxas or CUDA driver JIT)
layernorm.cubin
    ↓ (CuPy/CUDA driver)
Executable on GPU
```

### Python Integration
```python
# cutile_gpt_mlir/kernels.py
from cuda.cubin import load_cubin  # or custom loader

class MLIRKernelLoader:
    def load_kernel(self, cubin_path):
        # Load CUBIN or JIT compile bytecode
        pass

    def launch(self, grid, block, *args):
        # Launch kernel via CuPy or CUDA driver
        pass
```

## 📚 Documentation to Review

1. **CUDA Tile IR Spec**
   - https://docs.nvidia.com/cuda/tile-ir/13.1/
   - Focus on: Compilation Pipeline, Runtime

2. **MLIR GPU Dialect**
   - https://mlir.llvm.org/docs/Dialects/GPU/
   - Check: gpu-to-cubin pass

3. **NVPTX Backend**
   - https://llvm.org/docs/NVPTXUsage.html
   - Understand: LLVM → PTX lowering

## 🔍 Research Questions

1. Does CUDA Tile provide a backend compiler?
   - Check build artifacts in `build/cuda-tile/`
   - Look for tools like `cuda-tile-backend` or similar

2. Is bytecode executable directly?
   - Check if CUDA Tile has runtime library
   - Can we JIT compile .tilebc at runtime?

3. What's the official compilation flow?
   - Review CUDA Tile documentation
   - Check example projects in `external/cuda-tile/examples/`

## 💡 Current Best Guess

CUDA Tile is likely:
1. A **frontend** for writing GPU kernels in MLIR
2. Outputs **bytecode** as intermediate representation
3. Requires **separate backend** to generate PTX/CUBIN

Next investigation target:
```bash
# Check if there's a separate CUDA Tile backend
find external/cuda-tile -name "*backend*" -o -name "*codegen*"

# Or if NVVM conversion exists
grep -r "NVVM\|PTX" external/cuda-tile/lib/
```

## ⚡ Quick Win Option

If stuck on PTX generation, consider:
1. Use CUDA Tile for **high-level optimization** only
2. Lower to standard MLIR dialects (arith, func, etc.)
3. Use existing MLIR → NVVM → PTX pipeline
4. This would still benefit from Tile-level thinking

## 📞 Where to Get Help

1. CUDA Tile GitHub Issues
2. MLIR Discourse: https://discourse.llvm.org/c/mlir/
3. NVIDIA Developer Forums

---

**Status**: Infrastructure built ✅
**Blocker**: PTX/CUBIN generation pipeline unclear
**Priority**: Research CUDA Tile compilation backend
