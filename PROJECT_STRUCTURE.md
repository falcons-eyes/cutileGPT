# cutileGPT Project Structure

**최종 정리된 디렉토리 구조** (2026-01-26)

## 📊 개요

cutileGPT는 깔끔하게 정리된 디렉토리 구조로 **Tile Programming Philosophy**에 집중합니다.

```
cutileGPT/
├── 🎯 Core Implementation          # 핵심 구현
├── 📖 Documentation               # 문서
├── 🧪 Research & Experiments      # 연구/실험
├── 🔧 Tools & Scripts             # 도구/스크립트
└── 📊 Results & Logs              # 결과/로그
```

## 📁 상세 구조

### 루트 디렉토리 (핵심 파일만)

```
cutileGPT/
├── README.md                          # ⭐ 메인 문서 (Tile Philosophy 중심)
├── LICENSE                            # Apache-2.0 라이센스
├── pyproject.toml                     # 프로젝트 설정
├── uv.lock                            # 의존성 잠금 파일
│
├── demo_tile_gpt.py                   # 🎮 완전한 Tile Philosophy Demo
│
├── TILE_PHILOSOPHY_DEMO.md            # 📖 Tile Philosophy 완전 문서
├── ARCHITECTURE_VISION.md             # 🏗️ 프로젝트 비전 & 로드맵
├── CUTILE_PYTHON_PHILOSOPHY_ANALYSIS.md # 🔬 Philosophy 심층 분석
└── CHANGELOG_TILE_PHILOSOPHY.md       # 📝 Tile Philosophy 구현 히스토리
```

**특징**:
- ✅ 프로젝트 필수 파일만 유지
- ✅ Tile Philosophy 관련 핵심 문서
- ✅ 즉시 실행 가능한 demo

### 1️⃣ cutile_gpt/ - 핵심 구현

```
cutile_gpt/
├── model_tile.py                      # 🎯 Pure Tile Philosophy GPT
├── model.py                           # Original CuPy 기반 모델
├── compare.py                         # PyTorch vs cutileGPT 비교
│
└── kernels/                           # Declarative Tile Kernels
    ├── __init__.py
    ├── layernorm.py                   # ✅ Declarative normalization
    ├── gelu.py                        # ✅ 41x faster activation
    ├── linear.py                      # ✅ Tile-based matmul
    ├── linear_v2.py                   # Advanced features
    ├── attention.py                   # ✅ Flash Attention
    ├── attention_improved.py          # Improved version
    └── embedding.py                   # Embedding lookup
```

**역할**:
- Tile Programming Philosophy 실제 구현
- 모든 커널이 declarative 방식
- PyTorch 호환 모델

### 2️⃣ docs/ - 문서

```
docs/
├── OPTIMIZATION_SUMMARY.md            # 최적화 여정
├── PROFILING_SUMMARY.md               # 프로파일링 요약
├── VISUALIZATION_GUIDE.md             # 시각화 가이드
└── VISUALIZATION_SUMMARY.md           # 시각화 요약
```

**역할**:
- 성능 최적화 기록
- 프로파일링 결과 문서화
- 시각화 도구 사용법

### 3️⃣ scripts/ - 도구 & 스크립트

```
scripts/
├── run_nsys_profile.sh                # Nsight Systems 프로파일링
├── run_ncu_profile.sh                 # Nsight Compute 프로파일링
├── benchmark_tile_optimization.py     # Tile 최적화 벤치마크
├── profile_performance.py             # 성능 프로파일링
├── visualize_performance.py           # 성능 시각화
└── visualize_comparison.py            # 비교 시각화
```

**역할**:
- 프로파일링 자동화
- 성능 벤치마크
- 결과 시각화

### 4️⃣ tests/ - 테스트

```
tests/
├── test_text_generation.py            # 텍스트 생성 테스트
├── test_gpt2_real.py                  # GPT-2 실제 테스트
└── test_tile_sizes.py                 # Tile 크기 테스트
```

**역할**:
- 기능 테스트
- 정확성 검증
- 성능 테스트

### 5️⃣ mlir_research/ - MLIR 연구 (선택적)

```
mlir_research/
├── README.md                          # MLIR 연구 개요
├── LLVM_MLIR_BUILD_SOLUTION.md       # LLVM/MLIR 빌드 해결책
├── NEXT_STEPS.md                      # MLIR backend 다음 단계
├── GETTING_STARTED_MLIR.md            # MLIR 시작 가이드
├── CUDA_TILE_MLIR_INTEGRATION_ANALYSIS.md
├── CUDA_TILE_PHILOSOPHY_ANALYSIS.md
├── TILE_IR_EXPERIMENT_RESULTS.md
├── TILE_IR_IMPROVEMENTS.md
├── TILE_IR_SUMMARY_KR.md
│
├── setup_cuda_tile.sh                 # LLVM/MLIR 설치 스크립트
├── setup_cuda_tile_auto.sh
├── CMakeLists.txt                     # CMake 설정
├── cmake_*.log                        # 빌드 로그
│
├── cutile_gpt_mlir/                   # MLIR 커널 실험
│   ├── kernels/
│   │   ├── layernorm.mlir
│   │   └── test_simple.mlir
│   └── compiled/                      # 컴파일 출력
│
├── build/                             # LLVM/MLIR 빌드 (gitignore)
└── tools/                             # LLVM 도구 (gitignore)
```

**역할**:
- MLIR backend 연구 (선택적)
- Compile-time 최적화 탐구
- 메인 프로젝트와 분리된 실험

### 6️⃣ profiling_results/ - 성능 결과

```
profiling_results/
├── performance_dashboard.html         # 📊 대화형 대시보드
├── profiling_data.json                # 벤치마크 데이터
└── cutile_nsys.nsys-rep              # Nsight Systems 결과
```

**역할**:
- 성능 벤치마크 결과
- 프로파일링 데이터
- 시각화 대시보드

### 7️⃣ logs/ - 로그 파일

```
logs/
├── gpt2_test_output.txt              # GPT-2 테스트 출력
└── nsys_profile_log.txt              # 프로파일링 로그
```

**역할**:
- 테스트 출력 로그
- 프로파일링 로그
- 디버깅 정보

### 8️⃣ external/ - 외부 의존성

```
external/
├── cutile-python/                     # NVIDIA CUDA Tile (submodule)
└── minGPT/                           # Reference implementation (submodule)
```

**역할**:
- Git submodules
- 외부 라이브러리

## 🎯 파일 분류

### ✅ 유지해야 할 파일 (루트)

**프로젝트 필수**:
- README.md
- LICENSE
- pyproject.toml
- uv.lock

**Tile Philosophy 핵심**:
- demo_tile_gpt.py
- TILE_PHILOSOPHY_DEMO.md
- ARCHITECTURE_VISION.md
- CUTILE_PYTHON_PHILOSOPHY_ANALYSIS.md
- CHANGELOG_TILE_PHILOSOPHY.md

### 📂 정리된 위치

| 파일 유형 | 위치 |
|---------|------|
| 핵심 구현 | `cutile_gpt/` |
| Tile Philosophy 문서 | 루트 (5개 핵심 문서) |
| 최적화/프로파일링 문서 | `docs/` |
| 스크립트 | `scripts/` |
| 테스트 | `tests/` |
| MLIR 연구 | `mlir_research/` |
| 성능 결과 | `profiling_results/` |
| 로그 | `logs/` |
| 외부 라이브러리 | `external/` |

## 📊 디렉토리별 역할

### 메인 워크플로우

```
1. README.md 읽기
   ↓
2. demo_tile_gpt.py 실행
   ↓
3. cutile_gpt/ 커널 탐색
   ↓
4. TILE_PHILOSOPHY_DEMO.md로 깊이 이해
```

### 개발 워크플로우

```
1. cutile_gpt/에서 코드 작성
   ↓
2. tests/로 테스트
   ↓
3. scripts/로 프로파일링
   ↓
4. profiling_results/에서 결과 확인
```

### 연구 워크플로우

```
1. mlir_research/에서 MLIR 실험
   ↓
2. 빌드 & 컴파일
   ↓
3. 성능 비교
   ↓
4. docs/에 결과 문서화
```

## 🧹 정리 원칙

### ✅ 루트는 깔끔하게
- 프로젝트 필수 파일만
- 핵심 문서 5개만
- 즉시 실행 가능한 demo만

### ✅ 기능별로 분류
- 구현 → `cutile_gpt/`
- 문서 → `docs/`
- 스크립트 → `scripts/`
- 테스트 → `tests/`
- 연구 → `mlir_research/`

### ✅ 실용성 우선
- MLIR은 선택적 연구
- Python API가 메인
- Tile Philosophy 강조

## 🎓 핵심 메시지

cutileGPT의 디렉토리 구조는 다음을 명확히 전달합니다:

1. **Tile Programming Philosophy가 중심**
   - 루트의 핵심 문서들
   - demo_tile_gpt.py가 즉시 실행 가능

2. **실용성 우선**
   - cutile_gpt/ Python 구현이 메인
   - MLIR은 mlir_research/로 분리

3. **깔끔한 구조**
   - 역할별로 명확히 분류
   - 루트는 필수 파일만

4. **쉬운 탐색**
   - README.md → demo → 커널 → 문서
   - 명확한 경로

---

**이 구조는 Tile Programming Philosophy를 중심으로 최적화되었습니다!** 🚀

*Think in WHAT (operations), not HOW (threads)*
