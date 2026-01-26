# LLVM/MLIR Build Solution

## 문제 요약

LLVM 프로젝트에서 MLIR을 활성화했음에도 불구하고 MLIR 라이브러리가 빌드되지 않는 문제가 발생했습니다.

### 증상
- CMake 로그: "mlir project is enabled" 표시됨
- 하지만 빌드 후: MLIR 라이브러리 0개 생성
- `tools/mlir/lib/` 디렉토리가 생성되지 않음
- `find_package(MLIR)` 실패

## 근본 원인

CMake의 `-S`와 `-B` 플래그를 사용한 out-of-source 빌드 방식이 MLIR 하위 디렉토리를 제대로 처리하지 못했습니다.

### 작동하지 않은 방법 ❌
```bash
cd /some/directory
cmake -G Ninja \
    -S /path/to/llvm-project/llvm \
    -B /path/to/build/llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    ...
```

## 해결 방법 ✅

**빌드 디렉토리 내부에서 전통적인 CMake 방식 사용**

```bash
# 1. 빌드 디렉토리 생성 및 이동
mkdir -p /home/hwoo_joo/github/cutileGPT/build/llvm
cd /home/hwoo_joo/github/cutileGPT/build/llvm

# 2. CMake 설정 (빌드 디렉토리 내부에서 실행)
cmake ../../external/llvm-project/llvm \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../../tools/llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="NVPTX;X86" \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_EH=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON

# 3. 타겟 확인
ninja -t targets | grep "libMLIR" | wc -l
# 출력: 409 (성공!)

# 4. 빌드 (1-1.5시간 소요)
ninja -j$(nproc)

# 5. 설치
ninja install
```

## 핵심 차이점

### 작동하지 않은 방식
```bash
cmake -G Ninja -S <source> -B <build> ...
```
- `-S`와 `-B`를 사용한 경로 지정
- MLIR 하위 프로젝트가 제대로 포함되지 않음

### 작동한 방식
```bash
cd <build>
cmake <source> -G Ninja ...
```
- 빌드 디렉토리에서 직접 실행
- 상대 경로로 소스 디렉토리 지정
- MLIR 하위 프로젝트가 정상적으로 포함됨

## 검증 방법

### 1. CMake 설정 후 타겟 확인
```bash
cd build/llvm
ninja -t targets | grep "libMLIR" | wc -l
```
**기대 결과**: 409개 이상의 MLIR 라이브러리 타겟

### 2. 빌드 후 라이브러리 확인
```bash
find build/llvm/lib -name "libMLIR*.a" | wc -l
```
**기대 결과**: 400개 이상의 라이브러리 파일

### 3. 설치 후 도구 확인
```bash
ls tools/llvm/bin/mlir-*
```
**기대 결과**:
- mlir-opt
- mlir-translate
- mlir-tblgen
- 기타 MLIR 도구들

### 4. MLIR 패키지 확인
```bash
ls tools/llvm/lib/cmake/mlir/
```
**기대 결과**:
- MLIRConfig.cmake
- MLIRTargets.cmake

## 완전한 빌드 스크립트

```bash
#!/bin/bash
set -e

PROJECT_ROOT="/home/hwoo_joo/github/cutileGPT"
LLVM_SRC="${PROJECT_ROOT}/external/llvm-project"
LLVM_BUILD="${PROJECT_ROOT}/build/llvm"
LLVM_INSTALL="${PROJECT_ROOT}/tools/llvm"

# 클린 빌드
rm -rf "${LLVM_BUILD}" "${LLVM_INSTALL}"
mkdir -p "${LLVM_BUILD}"

# CMake 설정
cd "${LLVM_BUILD}"
cmake "${LLVM_SRC}/llvm" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL}" \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="NVPTX;X86" \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_EH=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_OPTIMIZED_TABLEGEN=ON

# 타겟 검증
MLIR_TARGETS=$(ninja -t targets | grep -c "libMLIR" || true)
echo "MLIR library targets: ${MLIR_TARGETS}"
if [ "${MLIR_TARGETS}" -lt 100 ]; then
    echo "ERROR: Expected at least 100 MLIR targets, got ${MLIR_TARGETS}"
    exit 1
fi

# 빌드
ninja -j$(nproc)

# 설치
ninja install

# 검증
if [ ! -f "${LLVM_INSTALL}/bin/mlir-opt" ]; then
    echo "ERROR: mlir-opt not found after installation"
    exit 1
fi

echo "LLVM/MLIR build and installation successful!"
echo "MLIR tools installed to: ${LLVM_INSTALL}/bin/"
```

## 주의사항

1. **절대 `-S`와 `-B` 플래그를 함께 사용하지 마세요**
   - 특히 LLVM 같은 복잡한 monorepo 프로젝트에서

2. **항상 빌드 디렉토리에서 CMake 실행**
   ```bash
   cd build/llvm
   cmake ../../external/llvm-project/llvm ...
   ```

3. **설정 후 타겟 수 확인**
   ```bash
   ninja -t targets | grep "libMLIR" | wc -l
   ```
   - 0이 나오면 설정이 잘못된 것

4. **CMake 캐시 문제 시 완전 삭제**
   ```bash
   rm -rf build/llvm
   ```

## 타임라인

- **설정**: ~10초
- **빌드**: 1-1.5시간 (머신 스펙에 따라 다름)
- **설치**: 5-10분

## 참고

이 솔루션은 LLVM commit `3d7018c70b97e6a3d6dfe08e9f11dede96242d1f`에서 테스트되었습니다.
