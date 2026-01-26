# cutileGPT Project Structure

**Final organized directory structure** (2026-01-26)

## 📊 Overview

cutileGPT has a clean directory structure focused on **Tile Programming Philosophy**.

```
cutileGPT/
├── 🎯 Core Implementation          # Core implementation
├── 📖 Documentation               # Documentation
├── 🧪 Research & Experiments      # Research/experiments
├── 🔧 Tools & Scripts             # Tools/scripts
└── 📊 Results & Logs              # Results/logs
```

## 📁 Detailed Structure

### Root Directory (Essential Files Only)

```
cutileGPT/
├── README.md                          # ⭐ Main documentation (Tile Philosophy focused)
├── LICENSE                            # Apache-2.0 License
├── pyproject.toml                     # Project configuration
├── uv.lock                            # Dependency lock file
│
└── demo_tile_gpt.py                   # 🎮 Complete Tile Philosophy Demo
```

**Features**:
- ✅ Only essential project files
- ✅ Core Tile Philosophy documentation
- ✅ Immediately executable demo

### 1️⃣ cutile_gpt/ - Core Implementation

```
cutile_gpt/
├── model_tile.py                      # 🎯 Pure Tile Philosophy GPT
├── model.py                           # Original CuPy-based model
├── compare.py                         # PyTorch vs cutileGPT comparison
│
└── kernels/                           # Declarative Tile Kernels
    ├── __init__.py
    ├── layernorm.py                   # ✅ Declarative normalization
    ├── gelu.py                        # ✅ 8.3x faster activation
    ├── linear.py                      # ✅ Tile-based matmul
    ├── linear_v2.py                   # Advanced features
    ├── attention.py                   # ✅ Flash Attention
    ├── attention_improved.py          # Improved version
    └── embedding.py                   # Embedding lookup
```

**Role**:
- Actual implementation of Tile Programming Philosophy
- All kernels follow declarative approach
- PyTorch-compatible model

### 2️⃣ docs/ - Documentation

```
docs/
├── TILE_PHILOSOPHY_DEMO.md            # Philosophy documentation
├── ARCHITECTURE_VISION.md             # Project vision & roadmap
├── CUTILE_PYTHON_PHILOSOPHY_ANALYSIS.md # Philosophy analysis
├── PROJECT_STRUCTURE.md               # This file
├── OPTIMIZATION_SUMMARY.md            # Optimization journey
├── PROFILING_SUMMARY.md               # Profiling summary
├── VISUALIZATION_GUIDE.md             # Visualization guide
└── VISUALIZATION_SUMMARY.md           # Visualization summary
```

**Role**:
- Performance optimization records
- Profiling result documentation
- Visualization tool usage

### 3️⃣ scripts/ - Tools & Scripts

```
scripts/
├── run_nsys_profile.sh                # Nsight Systems profiling
├── run_ncu_profile.sh                 # Nsight Compute profiling
├── benchmark_tile_optimization.py     # Tile optimization benchmark
├── profile_performance.py             # Performance profiling
├── visualize_performance.py           # Performance visualization
└── visualize_comparison.py            # Comparison visualization
```

**Role**:
- Profiling automation
- Performance benchmarking
- Result visualization

### 4️⃣ tests/ - Tests

```
tests/
├── test_text_generation.py            # Text generation tests
├── test_gpt2_real.py                  # Real GPT-2 tests
└── test_tile_sizes.py                 # Tile size tests
```

**Role**:
- Functional testing
- Correctness validation
- Performance testing

### 5️⃣ mlir_research/ - MLIR Research (Optional)

```
mlir_research/
├── README.md                          # MLIR research overview
├── LLVM_MLIR_BUILD_SOLUTION.md       # LLVM/MLIR build solution
├── NEXT_STEPS.md                      # MLIR backend next steps
├── GETTING_STARTED_MLIR.md            # MLIR getting started
├── CUDA_TILE_MLIR_INTEGRATION_ANALYSIS.md
├── CUDA_TILE_PHILOSOPHY_ANALYSIS.md
├── TILE_IR_EXPERIMENT_RESULTS.md
├── TILE_IR_IMPROVEMENTS.md
├── TILE_IR_SUMMARY_KR.md
│
├── setup_cuda_tile.sh                 # LLVM/MLIR installation script
├── setup_cuda_tile_auto.sh
├── CMakeLists.txt                     # CMake configuration
├── cmake_*.log                        # Build logs
│
├── cutile_gpt_mlir/                   # MLIR kernel experiments
│   ├── kernels/
│   │   ├── layernorm.mlir
│   │   └── test_simple.mlir
│   └── compiled/                      # Compiled output
│
├── build/                             # LLVM/MLIR build (gitignore)
└── tools/                             # LLVM tools (gitignore)
```

**Role**:
- MLIR backend research (optional)
- Compile-time optimization exploration
- Separated experiments from main project

### 6️⃣ profiling_results/ - Performance Results

```
profiling_results/
├── performance_dashboard.html         # 📊 Interactive dashboard
├── profiling_data.json                # Benchmark data
└── cutile_nsys.nsys-rep              # Nsight Systems results
```

**Role**:
- Performance benchmark results
- Profiling data
- Visualization dashboard

### 7️⃣ logs/ - Log Files

```
logs/
├── gpt2_test_output.txt              # GPT-2 test output
└── nsys_profile_log.txt              # Profiling logs
```

**Role**:
- Test output logs
- Profiling logs
- Debugging information

### 8️⃣ external/ - External Dependencies

```
external/
├── cutile-python/                     # NVIDIA CUDA Tile (submodule)
└── minGPT/                           # Reference implementation (submodule)
```

**Role**:
- Git submodules
- External libraries

## 🎯 File Classification

### ✅ Files to Keep (Root)

**Project Essentials**:
- README.md
- LICENSE
- pyproject.toml
- uv.lock

**Tile Philosophy Core**:
- demo_tile_gpt.py

### 📂 Organized Location

| File Type | Location |
|---------|---------|
| Core implementation | `cutile_gpt/` |
| Tile Philosophy docs | `docs/` |
| Optimization/profiling docs | `docs/` |
| Scripts | `scripts/` |
| Tests | `tests/` |
| MLIR research | `mlir_research/` |
| Performance results | `profiling_results/` |
| Logs | `logs/` |
| External libraries | `external/` |

## 📊 Directory Roles

### Main Workflow

```
1. Read README.md
   ↓
2. Run demo_tile_gpt.py
   ↓
3. Explore cutile_gpt/ kernels
   ↓
4. Deep dive with docs/TILE_PHILOSOPHY_DEMO.md
```

### Development Workflow

```
1. Write code in cutile_gpt/
   ↓
2. Test with tests/
   ↓
3. Profile with scripts/
   ↓
4. Check results in profiling_results/
```

### Research Workflow

```
1. MLIR experiments in mlir_research/
   ↓
2. Build & compile
   ↓
3. Performance comparison
   ↓
4. Document results in docs/
```

## 🧹 Organization Principles

### ✅ Keep Root Clean
- Only essential project files
- Only core documentation
- Only immediately executable demo

### ✅ Classify by Function
- Implementation → `cutile_gpt/`
- Documentation → `docs/`
- Scripts → `scripts/`
- Tests → `tests/`
- Research → `mlir_research/`

### ✅ Practicality First
- MLIR is optional research
- Python API is main focus
- Tile Philosophy emphasized

## 🎓 Core Message

cutileGPT's directory structure clearly conveys:

1. **Tile Programming Philosophy is Central**
   - Core documentation in docs/
   - demo_tile_gpt.py immediately executable

2. **Practicality First**
   - cutile_gpt/ Python implementation is main
   - MLIR separated to mlir_research/

3. **Clean Structure**
   - Clearly classified by role
   - Root contains only essentials

4. **Easy Navigation**
   - README.md → demo → kernels → docs
   - Clear path

---

**This structure is optimized around Tile Programming Philosophy!** 🚀

*Think in WHAT (operations), not HOW (threads)*
