# CUDA Tile MLIR 통합 가능성 분석

## 🎯 목표

NVIDIA의 공식 `cuda-tile` 레포지토리를 사용하여 진짜 Tile 철학을 cutileGPT에 적용할 수 있는지 검토합니다.

---

## 📦 cuda-tile 레포지토리 구조

### 추가 완료
```bash
git submodule add https://github.com/NVIDIA/cuda-tile.git external/cuda-tile
```

### 핵심 컴포넌트

1. **CUDA Tile Dialect** (MLIR)
   - Tile 기반 연산의 first-class operation/type
   - MLIR IR 표현

2. **Python Bindings**
   - Python에서 MLIR IR 조작 가능
   - But: 커널 작성 X, IR 조작만

3. **Bytecode System**
   - MLIR → Bytecode → Cubin
   - 직접 CUDA Driver API로 로드 가능

4. **Conformance Tests**
   - 다양한 MLIR 예제
   - Operation 사용법 참고

---

## 🔧 사전 설치 요구사항

### 필수 요구사항

```bash
# 1. Build Tools
- CMake 3.20.0+
- C++17 compatible compiler (GCC 9+, Clang 10+)
- Ninja build system

# 2. MLIR/LLVM
- Specific LLVM commit (자동 다운로드 or 수동 빌드)
- MLIR Python bindings (optional)

# 3. CUDA
- CUDA Toolkit 13.1+ (for tileiras compiler)
- Compatible GPU (sm_80+, Ampere/Hopper/Blackwell)
- CUDA Driver API support

# 4. Python
- Python 3.6+ (for bindings)
```

### 빌드 시간 예상
```
Automatic LLVM download + build: ~1-2 hours (first time)
CUDA Tile build: ~10-20 minutes
Total: ~1.5-2.5 hours
```

---

## 🚀 빌드 과정

### Option 1: Quick Start (자동 LLVM 다운로드)

```bash
cd external/cuda-tile

# Configure
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCUDA_TILE_ENABLE_BINDINGS_PYTHON=ON

# Build (시간 오래 걸림!)
cmake --build build

# Test
cmake --build build --target check-cuda-tile

# Install tools
cmake --install build --prefix ../../tools/cuda-tile
```

### Option 2: Pre-built LLVM 사용 (빠름)

```bash
# 1. LLVM 미리 빌드 (한 번만)
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout <compatible-commit>  # cuda-tile/cmake/IncludeLLVM.cmake 참고

cmake -G Ninja -S llvm -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="NVPTX;X86"

cmake --build build
cmake --install build --prefix /opt/llvm

# 2. CUDA Tile 빌드 (빠름)
cd ../cuda-tile
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_TILE_USE_LLVM_INSTALL_DIR=/opt/llvm \
  -DCUDA_TILE_ENABLE_BINDINGS_PYTHON=ON

cmake --build build
```

---

## 📝 MLIR로 커널 작성하기

### 예제: 간단한 Vector Add

```mlir
// vector_add.mlir
cuda_tile.module @vector_add_module {
    cuda_tile.entry @vector_add(
        %a_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %b_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %c_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>
    ) {
        // Load tiles
        %token0 = make_token : !cuda_tile.token
        %a, %token1 = load_ptr_tko weak %a_ptr token=%token0
            : !cuda_tile.tile<!cuda_tile.ptr<f32>> -> !cuda_tile.tile<128xf32>, !cuda_tile.token

        %b, %token2 = load_ptr_tko weak %b_ptr token=%token1
            : !cuda_tile.tile<!cuda_tile.ptr<f32>> -> !cuda_tile.tile<128xf32>, !cuda_tile.token

        // Add tiles
        %c = addf %a, %b : !cuda_tile.tile<128xf32>

        // Store result
        %token3 = store_ptr_tko weak %c_ptr, %c token=%token2
            : !cuda_tile.tile<!cuda_tile.ptr<f32>>, !cuda_tile.tile<128xf32> -> !cuda_tile.token

        return
    }
}
```

### 예제: Matrix Multiply (진짜 Tile 스타일!)

```mlir
// matmul.mlir
cuda_tile.module @matmul_module {
    cuda_tile.entry @matmul(
        %A_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %B_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %C_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %M: !cuda_tile.tile<i32>,
        %N: !cuda_tile.tile<i32>,
        %K: !cuda_tile.tile<i32>
    ) {
        // Constants
        %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
        %c1 = constant <i32: 1> : !cuda_tile.tile<i32>
        %c_tile_size = constant <i32: 32> : !cuda_tile.tile<i32>

        // Initialize accumulator
        %zero_f32 = constant <f32: 0.0> : !cuda_tile.tile<f32>
        %acc_init = broadcast %zero_f32 : !cuda_tile.tile<f32> -> !cuda_tile.tile<32x32xf32>

        // K-dimension loop (reduction)
        %final_acc = for %k_idx in (%c0 to %K, step %c_tile_size) : tile<i32>
            iter_values(%acc = %acc_init) -> (tile<32x32xf32>)
        {
            // Load A tile (32x32)
            %A_tile, %token_a = load_ptr_tko weak %A_ptr
                : !cuda_tile.tile<!cuda_tile.ptr<f32>> -> !cuda_tile.tile<32x32xf32>, !cuda_tile.token

            // Load B tile (32x32)
            %B_tile, %token_b = load_ptr_tko weak %B_ptr token=%token_a
                : !cuda_tile.tile<!cuda_tile.ptr<f32>> -> !cuda_tile.tile<32x32xf32>, !cuda_tile.token

            // Matrix multiply-accumulate
            %new_acc = mmaf %A_tile, %B_tile, %acc
                : tile<32x32xf32>, tile<32x32xf32>, tile<32x32xf32>

            continue %new_acc : tile<32x32xf32>
        }

        // Store result
        %token_out = store_ptr_tko weak %C_ptr, %final_acc
            : !cuda_tile.tile<!cuda_tile.ptr<f32>>, !cuda_tile.tile<32x32xf32> -> !cuda_tile.token

        return
    }
}
```

### 주목할 점: 진짜 선언적!

```mlir
// ❌ Python API (현재 cutileGPT)
bid_m, bid_n = swizzle_2d(M, N, tm, tn)  // 수동 인덱싱
offs_m = bid_x * TILE_M + ct.arange()    // 수동 오프셋

// ✅ MLIR (진짜 Tile 스타일)
%final_acc = for %k_idx in (%c0 to %K, step %c_tile_size) : tile<i32>
    iter_values(%acc = %acc_init) -> (tile<32x32xf32>)
{
    // 컴파일러가 알아서:
    // - 블록 인덱싱
    // - 메모리 레이아웃
    // - 텐서 코어 매핑
}
```

---

## 🔄 컴파일 및 실행 플로우

### Step 1: MLIR → Bytecode

```bash
cuda-tile-translate vector_add.mlir \
    --bytecode-version=13.1 \
    --mlir-to-cudatilebc \
    --no-implicit-module \
    -o vector_add.tilebc
```

### Step 2: Bytecode → Cubin (AoT compilation)

```bash
# CUDA Toolkit의 tileiras 사용
tileiras --gpu-name sm_100 vector_add.tilebc -o vector_add.cubin
```

또는 JIT compilation:
```cpp
// Bytecode를 직접 로드 (JIT)
cuModuleLoad(&module, "vector_add.tilebc");
```

### Step 3: C++/Python에서 실행

```cpp
// C++ (CUDA Driver API)
CUmodule module;
CUfunction kernel;

cuModuleLoad(&module, "vector_add.cubin");
cuModuleGetFunction(&kernel, module, "vector_add");

void* args[] = {&a_ptr, &b_ptr, &c_ptr};
cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, stream, args, NULL);
```

```python
# Python (cupy)
import cupy as cp
from cupy.cuda import driver

module = driver.moduleLoad("vector_add.cubin")
kernel = module.get_function("vector_add")

kernel.launch(grid=(1,1,1), block=(1,1,1), args=[a_ptr, b_ptr, c_ptr])
```

---

## ✅ Tile 철학 적용 가능성

### 1. 진짜 Tile 스타일 가능! ✅

**MLIR로 작성하면:**
- ✅ 선언적 프로그래밍
- ✅ 컴파일러 주도 최적화
- ✅ 하드웨어 추상화
- ✅ 블록 인덱싱 자동
- ✅ Loop-carried values 명시적 (`iter_values`)

**예시:**
```mlir
// 선언적: "무엇을" 계산할지만
%result = for %i in (%start to %end, step %step) : tile<i32>
    iter_values(%acc = %init) -> (tile<32x32xf32>)
{
    %a = load_ptr_tko weak %a_ptr : ... -> tile<32x32xf32>, token
    %b = load_ptr_tko weak %b_ptr : ... -> tile<32x32xf32>, token
    %new_acc = mmaf %a, %b, %acc : tile<32x32xf32>, ...
    continue %new_acc : tile<32x32xf32>
}
```

### 2. 통합 방법

#### Option A: MLIR 커널 + Python 호스트 (추천) ✅

```
cutileGPT/
├── kernels_mlir/
│   ├── attention.mlir       # MLIR로 작성
│   ├── linear.mlir
│   └── layernorm.mlir
├── kernels_compiled/
│   ├── attention.tilebc     # 컴파일된 bytecode
│   └── attention.cubin      # AoT 컴파일 (optional)
└── cutile_gpt/
    └── model.py             # Python에서 로드
```

```python
# model.py
import cupy as cp
from cupy.cuda import driver

class CutileGPTMLIR:
    def __init__(self):
        # Load compiled kernels
        self.attention_module = driver.moduleLoad("kernels_compiled/attention.cubin")
        self.linear_module = driver.moduleLoad("kernels_compiled/linear.cubin")

        self.attention_kernel = self.attention_module.get_function("causal_attention")
        self.linear_kernel = self.linear_module.get_function("matmul")

    def forward(self, x):
        # Launch MLIR kernels from Python
        self.attention_kernel.launch(...)
        self.linear_kernel.launch(...)
```

**장점:**
- ✅ 진짜 Tile 철학 적용
- ✅ Python 인터페이스 유지
- ✅ 교육적 가치 극대화

**단점:**
- ⚠️ MLIR 학습 필요
- ⚠️ 빌드 복잡도 증가
- ⚠️ 디버깅 어려움

#### Option B: Hybrid (Python + MLIR) ⚠️

```python
# 간단한 것: Python API 유지
from cutile_gpt.kernels.linear import cutile_linear

# 복잡한 것: MLIR 커널 사용
attention_module = load_mlir_kernel("attention.cubin")
```

**장점:**
- ✅ 점진적 마이그레이션
- ✅ 복잡도 분산

**단점:**
- ⚠️ 두 시스템 유지보수

#### Option C: Pure MLIR ❌ 비추천

```mlir
// 전체를 MLIR로 재작성
cuda_tile.module @gpt_model { ... }
```

**이유:**
- ❌ Python 인터페이스 포기
- ❌ 유연성 감소
- ❌ 실용성 낮음

---

## 📊 현실적 평가

### 할 수 있는 것 ✅

1. **MLIR 커널 작성**
   - Attention, Linear, LayerNorm
   - 진짜 선언적 스타일
   - Tile 철학 100% 적용

2. **컴파일 및 실행**
   - MLIR → Bytecode → Cubin
   - Python/C++에서 로드
   - 성능은 비슷할 것 (같은 컴파일러)

3. **교육적 가치**
   - "이게 진짜 Tile 스타일이다" showcase
   - Python API vs MLIR 비교

### 해야 하는 것 ⚠️

1. **빌드 인프라**
   - LLVM/MLIR 빌드 (1-2시간)
   - CUDA Tile 빌드
   - CI/CD 통합

2. **MLIR 학습**
   - Operation semantics
   - Type system
   - Bytecode format

3. **디버깅 도구**
   - MLIR 디버깅 어려움
   - Bytecode 검증

### 얻는 것 vs 잃는 것

**얻는 것:**
- ✅ **진짜 Tile 철학** 적용
- ✅ **교육적 가치** 극대화
- ✅ **프로젝트 정체성** 명확화

**잃는 것:**
- ⚠️ **개발 속도** 감소
- ⚠️ **유지보수 복잡도** 증가
- ⚠️ **접근성** 감소 (MLIR 진입장벽)

---

## 💡 추천 방향

### Option A: "Dual Implementation" ✅ **강력 추천**

**구조:**
```
cutileGPT/
├── cutile_gpt/           # 현재 Python 구현 (유지)
│   └── kernels/
│       ├── linear.py     # "Tile API Tutorial"
│       └── attention.py
├── cutile_gpt_mlir/      # 새로운 MLIR 구현
│   ├── kernels/
│   │   ├── linear.mlir   # "True Tile Philosophy"
│   │   └── attention.mlir
│   └── model_mlir.py     # MLIR 커널 사용
└── docs/
    └── comparison.md     # Python vs MLIR 비교
```

**가치:**
1. **교육적**: 두 접근법 비교 가능
2. **실용적**: Python 버전은 쉽게 사용
3. **철학적**: MLIR 버전은 진짜 Tile 스타일

**README:**
```markdown
## Two Implementations

### 1. Python API (cutile_gpt/)
- 🎓 Educational: Learn Tile API usage
- 🚀 Practical: Easy to use and modify
- ⚠️ Note: Low-level optimizations included

### 2. MLIR (cutile_gpt_mlir/)
- 🏛️ Philosophical: True Tile-based thinking
- 📚 Advanced: Compiler-driven optimization
- 🎯 Showcase: What Tile IR should be
```

### Option B: "MLIR Only" ⚠️ 위험

전체를 MLIR로 재작성
- ❌ 접근성 크게 감소
- ❌ 현재 코드 폐기
- ❌ 실용성 희생

### Option C: "현재 유지" 😐 안전하지만 아쉬움

Python API만 유지
- ✅ 안전하고 실용적
- ❌ "Tile 철학" 문제 해결 안 됨
- ❌ 정체성 모호

---

## 🎯 결론 및 Next Steps

### 핵심 질문에 대한 답

**Q: MLIR로 진짜 Tile 철학 적용 가능한가?**
**A: 예! 100% 가능합니다.**

**Q: cutileGPT에 적용할 가치가 있는가?**
**A: 예, 하지만 "Dual Implementation" 형태로.**

### 추천 로드맵

#### Phase 1: 환경 구축 (1-2일)
```bash
# 1. LLVM/MLIR 빌드
# 2. CUDA Tile 빌드
# 3. 도구 설치 확인
```

#### Phase 2: 간단한 MLIR 커널 (3-5일)
```mlir
# 1. LayerNorm (가장 간단)
# 2. Linear (matmul)
# 3. 성능 비교 with Python version
```

#### Phase 3: Attention 구현 (5-7일)
```mlir
# 1. Flash Attention in MLIR
# 2. 진짜 선언적 스타일
# 3. 교육 자료 작성
```

#### Phase 4: 문서화 (3-5일)
```markdown
# 1. Python vs MLIR 비교 문서
# 2. "True Tile Philosophy" 가이드
# 3. MLIR 튜토리얼
```

### 최종 권장사항

**"Dual Implementation"으로 가세요!**

1. ✅ 현재 Python 코드 유지 (실용성)
2. ✅ MLIR 버전 추가 (철학)
3. ✅ 둘을 비교하는 문서 (교육)

이렇게 하면:
- **실용적 가치** 유지
- **교육적 가치** 극대화
- **프로젝트 정체성** 명확화
- **"이게 진짜 Tile 스타일"** 증명

**결과:**
cutileGPT = 가장 포괄적인 CUDA Tile 교육 자료 🎓

---

## 📚 참고 자료

- [cuda-tile GitHub](https://github.com/NVIDIA/cuda-tile)
- [CUDA Tile IR Specification](https://docs.nvidia.com/cuda/tile-ir/13.1/)
- [MLIR Documentation](https://mlir.llvm.org/)
- cutileGPT Analysis: [CUDA_TILE_PHILOSOPHY_ANALYSIS.md](CUDA_TILE_PHILOSOPHY_ANALYSIS.md)
