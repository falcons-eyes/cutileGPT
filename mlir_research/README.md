# MLIR Research (Optional Path)

이 디렉토리는 MLIR backend를 통한 compile-time 최적화 연구를 위한 것입니다.

**현재 상태**: 인프라 구축 완료, PTX 생성 파이프라인 조사 중

## ⚠️ 중요

**MLIR은 선택사항입니다!** cutileGPT의 Python API 구현이 이미 Tile Programming Philosophy를 완벽하게 따르고 있으며, 실용적인 성능을 제공합니다.

## 📁 파일 구조

```
mlir_research/
├── README.md                           # 이 문서
├── LLVM_MLIR_BUILD_SOLUTION.md        # LLVM/MLIR 빌드 해결 방법
├── NEXT_STEPS.md                      # MLIR backend 다음 단계
├── setup_cuda_tile.sh                 # LLVM/MLIR 설치 스크립트
├── cutile_gpt_mlir/                   # MLIR 커널 (실험적)
│   ├── kernels/
│   │   ├── layernorm.mlir
│   │   └── test_simple.mlir
│   └── compiled/                      # 컴파일 출력 대상
└── (build/, tools/, external/ - 필요시 설치)
```

## 🎯 연구 목적

MLIR을 통해 다음을 탐구합니다:
- Compile-time 최적화
- 하드웨어 독립적 중간 표현
- 자동 커널 튜닝
- Tile-level 최적화 패스

## 📊 현재 vs MLIR

| Aspect | Python API (현재) | MLIR (연구) |
|--------|------------------|-------------|
| **상태** | ✅ 완전 동작 | 🚧 조사 중 |
| **성능** | ✅ 41x speedup | ❓ TBD |
| **사용성** | ✅ 쉬움 | ❌ 복잡 |
| **이식성** | ✅ GPU 독립적 | ✅ 더 나음 |
| **개발 속도** | ✅ 빠름 | ⚠️ 느림 |

## 🚀 MLIR 빌드 (선택)

```bash
cd mlir_research
bash setup_cuda_tile.sh
```

빌드 시간: ~1.5시간 (LLVM/MLIR)

## 📝 참고 문서

- [LLVM_MLIR_BUILD_SOLUTION.md](LLVM_MLIR_BUILD_SOLUTION.md) - 빌드 이슈 해결
- [NEXT_STEPS.md](NEXT_STEPS.md) - PTX/CUBIN 생성 조사
- [cutile_gpt_mlir/](cutile_gpt_mlir/) - MLIR 커널 예제

## 🎓 학습 자료

- [CUDA Tile IR Spec](https://docs.nvidia.com/cuda/tile-ir/13.1/)
- [MLIR GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)
- [NVPTX Backend](https://llvm.org/docs/NVPTXUsage.html)

## ⏭️ 다음 단계

1. **PTX/CUBIN 생성 파이프라인 조사**
   - CUDA Tile bytecode → PTX 변환 방법
   - JIT vs AOT 컴파일 옵션

2. **Python 통합 레이어**
   - CUBIN 로더 구현
   - CuPy와의 통합

3. **성능 비교**
   - MLIR vs Python API 벤치마크

## 🔗 관련 프로젝트

- [NVIDIA cuda-tile](https://github.com/NVIDIA/cuda-tile)
- [LLVM/MLIR](https://github.com/llvm/llvm-project)

---

**결론**: Python API가 현재 실용적인 선택입니다. MLIR은 장기 연구 프로젝트입니다.
