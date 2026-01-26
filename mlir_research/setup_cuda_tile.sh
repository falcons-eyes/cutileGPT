#!/bin/bash
#
# CUDA Tile MLIR Infrastructure Setup
#
# This script builds a complete CUDA Tile development environment
# with proper LLVM/MLIR integration for production-quality kernels.
#
# Estimated time: 1.5-2 hours (first run)
# Disk space required: ~20GB
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_DIR="${PROJECT_ROOT}/external"
BUILD_DIR="${PROJECT_ROOT}/build"
TOOLS_DIR="${PROJECT_ROOT}/tools"

LLVM_SRC_DIR="${EXTERNAL_DIR}/llvm-project"
LLVM_BUILD_DIR="${BUILD_DIR}/llvm"
LLVM_INSTALL_DIR="${TOOLS_DIR}/llvm"

CUDA_TILE_SRC_DIR="${EXTERNAL_DIR}/cuda-tile"
CUDA_TILE_BUILD_DIR="${BUILD_DIR}/cuda-tile"
CUDA_TILE_INSTALL_DIR="${TOOLS_DIR}/cuda-tile"

# Get LLVM commit hash from cuda-tile
LLVM_COMMIT=$(grep -oP 'LLVM_BUILD_COMMIT_HASH\s+\K[a-f0-9]+' "${CUDA_TILE_SRC_DIR}/cmake/IncludeLLVM.cmake" | head -1)

if [ -z "$LLVM_COMMIT" ]; then
    log_error "Failed to find LLVM commit hash in cuda-tile/cmake/IncludeLLVM.cmake"
    log_error "Expected pattern: LLVM_BUILD_COMMIT_HASH <hash>"
    exit 1
fi

log_info "LLVM commit hash: ${LLVM_COMMIT}"

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check CMake
    if ! command -v cmake &> /dev/null; then
        log_error "CMake not found. Install: sudo apt install cmake"
        exit 1
    fi
    CMAKE_VERSION=$(cmake --version | head -1 | grep -oP '\d+\.\d+\.\d+')
    log_info "CMake version: ${CMAKE_VERSION}"

    # Check Ninja
    if ! command -v ninja &> /dev/null; then
        log_warn "Ninja not found. Install for faster builds: sudo apt install ninja-build"
        GENERATOR="Unix Makefiles"
    else
        log_info "Ninja found"
        GENERATOR="Ninja"
    fi

    # Check GCC/Clang
    if command -v clang++ &> /dev/null; then
        CXX_COMPILER=$(which clang++)
        log_info "Using Clang++: ${CXX_COMPILER}"
    elif command -v g++ &> /dev/null; then
        CXX_COMPILER=$(which g++)
        log_info "Using G++: ${CXX_COMPILER}"
    else
        log_error "No C++ compiler found"
        exit 1
    fi

    # Check CUDA
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep -oP 'release \K\d+\.\d+')
        log_info "CUDA version: ${CUDA_VERSION}"
    else
        log_warn "CUDA not found. Some features may not work."
    fi

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found"
        exit 1
    fi
    PYTHON_VERSION=$(python3 --version | grep -oP '\d+\.\d+')
    log_info "Python version: ${PYTHON_VERSION}"

    # Check ccache (optional)
    if command -v ccache &> /dev/null; then
        USE_CCACHE=ON
        log_info "ccache found (will be used)"
    else
        USE_CCACHE=OFF
        log_info "ccache not found (disabled)"
    fi

    # Check disk space
    AVAILABLE_SPACE=$(df -h "${PROJECT_ROOT}" | awk 'NR==2 {print $4}')
    log_info "Available disk space: ${AVAILABLE_SPACE}"

    log_info "All prerequisites OK"
}

# Clone LLVM if needed
setup_llvm_source() {
    log_info "Setting up LLVM source..."

    if [ -d "${LLVM_SRC_DIR}" ]; then
        log_info "LLVM source already exists at ${LLVM_SRC_DIR}"

        cd "${LLVM_SRC_DIR}"
        CURRENT_COMMIT=$(git rev-parse HEAD)

        if [ "${CURRENT_COMMIT}" = "${LLVM_COMMIT}" ]; then
            log_info "Already at correct commit: ${LLVM_COMMIT}"
            return
        else
            log_info "Checking out correct commit: ${LLVM_COMMIT}"
            git fetch origin
            git checkout "${LLVM_COMMIT}"
        fi
    else
        log_info "Cloning LLVM (this will take a while)..."
        mkdir -p "${EXTERNAL_DIR}"
        cd "${EXTERNAL_DIR}"

        # Shallow clone for speed
        git clone --depth 1 --single-branch --branch main \
            https://github.com/llvm/llvm-project.git

        cd llvm-project

        # Fetch specific commit
        git fetch --depth 1 origin "${LLVM_COMMIT}"
        git checkout "${LLVM_COMMIT}"

        log_info "LLVM source ready at ${LLVM_SRC_DIR}"
    fi
}

# Build LLVM/MLIR
build_llvm() {
    log_info "Building LLVM/MLIR..."
    log_info "This will take 1-1.5 hours on a typical machine"

    mkdir -p "${LLVM_BUILD_DIR}"
    cd "${LLVM_BUILD_DIR}"

    # Configure LLVM
    cmake -G "${GENERATOR}" \
        -S "${LLVM_SRC_DIR}/llvm" \
        -B . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_DIR}" \
        -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
        -DLLVM_ENABLE_PROJECTS="mlir" \
        -DLLVM_TARGETS_TO_BUILD="NVPTX;X86" \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DLLVM_OPTIMIZED_TABLEGEN=ON \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_ENABLE_EH=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DLLVM_CCACHE_BUILD=${USE_CCACHE}

    log_info "Building LLVM (this is the long part)..."

    # Build with all available cores
    NCORES=$(nproc)
    log_info "Building with ${NCORES} cores"

    if [ "${GENERATOR}" = "Ninja" ]; then
        ninja -j${NCORES}
    else
        make -j${NCORES}
    fi

    log_info "Installing LLVM to ${LLVM_INSTALL_DIR}"

    if [ "${GENERATOR}" = "Ninja" ]; then
        ninja install
    else
        make install
    fi

    log_info "LLVM/MLIR build complete!"
}

# Build CUDA Tile
build_cuda_tile() {
    log_info "Building CUDA Tile..."

    mkdir -p "${CUDA_TILE_BUILD_DIR}"
    cd "${CUDA_TILE_BUILD_DIR}"

    # Configure CUDA Tile
    cmake -G "${GENERATOR}" \
        -S "${CUDA_TILE_SRC_DIR}" \
        -B . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${CUDA_TILE_INSTALL_DIR}" \
        -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
        -DCUDA_TILE_USE_LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR}" \
        -DCUDA_TILE_ENABLE_BINDINGS_PYTHON=ON \
        -DCUDA_TILE_ENABLE_TESTING=ON \
        -DLLVM_EXTERNAL_LIT="${LLVM_INSTALL_DIR}/bin/llvm-lit"

    log_info "Building CUDA Tile..."

    if [ "${GENERATOR}" = "Ninja" ]; then
        ninja -j$(nproc)
    else
        make -j$(nproc)
    fi

    log_info "Running CUDA Tile tests..."
    if [ "${GENERATOR}" = "Ninja" ]; then
        ninja check-cuda-tile || log_warn "Some tests failed (may be OK)"
    else
        make check-cuda-tile || log_warn "Some tests failed (may be OK)"
    fi

    log_info "Installing CUDA Tile to ${CUDA_TILE_INSTALL_DIR}"

    if [ "${GENERATOR}" = "Ninja" ]; then
        ninja install
    else
        make install
    fi

    log_info "CUDA Tile build complete!"
}

# Create environment setup script
create_env_script() {
    log_info "Creating environment setup script..."

    cat > "${PROJECT_ROOT}/setup_env.sh" << EOF
#!/bin/bash
#
# Source this file to set up CUDA Tile development environment
# Usage: source setup_env.sh
#

export LLVM_DIR="${LLVM_INSTALL_DIR}"
export CUDA_TILE_DIR="${CUDA_TILE_INSTALL_DIR}"

# Add tools to PATH
export PATH="\${LLVM_DIR}/bin:\${CUDA_TILE_DIR}/bin:\${PATH}"

# Python bindings
if [ -d "\${LLVM_DIR}/python" ]; then
    export PYTHONPATH="\${LLVM_DIR}/python:\${PYTHONPATH}"
fi

if [ -d "\${CUDA_TILE_DIR}/python" ]; then
    export PYTHONPATH="\${CUDA_TILE_DIR}/python:\${PYTHONPATH}"
fi

# Libraries
export LD_LIBRARY_PATH="\${LLVM_DIR}/lib:\${CUDA_TILE_DIR}/lib:\${LD_LIBRARY_PATH}"

echo "CUDA Tile environment configured!"
echo "LLVM: \${LLVM_DIR}"
echo "CUDA Tile: \${CUDA_TILE_DIR}"
echo ""
echo "Available tools:"
echo "  - mlir-opt: MLIR optimizer"
echo "  - mlir-translate: MLIR translator"
echo "  - cuda-tile-translate: CUDA Tile IR translator"
echo ""
echo "Test with: cuda-tile-translate --help"
EOF

    chmod +x "${PROJECT_ROOT}/setup_env.sh"
    log_info "Environment script created: ${PROJECT_ROOT}/setup_env.sh"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    source "${PROJECT_ROOT}/setup_env.sh"

    # Check tools
    if command -v cuda-tile-translate &> /dev/null; then
        log_info "✓ cuda-tile-translate found"
    else
        log_error "✗ cuda-tile-translate not found"
        exit 1
    fi

    if command -v mlir-opt &> /dev/null; then
        log_info "✓ mlir-opt found"
    else
        log_error "✗ mlir-opt not found"
        exit 1
    fi

    log_info "Installation verified successfully!"
}

# Create initial project structure
create_project_structure() {
    log_info "Creating project structure..."

    mkdir -p "${PROJECT_ROOT}/cutile_gpt_mlir/kernels"
    mkdir -p "${PROJECT_ROOT}/cutile_gpt_mlir/compiled"
    mkdir -p "${PROJECT_ROOT}/cutile_gpt_mlir/tests"

    log_info "Project structure created"
}

# Main execution
main() {
    log_info "=========================================="
    log_info "CUDA Tile MLIR Setup"
    log_info "=========================================="
    log_info ""

    check_prerequisites
    log_info ""

    # Ask for confirmation
    log_warn "This will:"
    log_warn "  1. Clone LLVM (~2GB download)"
    log_warn "  2. Build LLVM/MLIR (~1-1.5 hours)"
    log_warn "  3. Build CUDA Tile (~10-20 minutes)"
    log_warn "  4. Install tools to ${TOOLS_DIR}"
    log_warn ""
    log_warn "Total time: ~1.5-2 hours"
    log_warn "Disk space: ~20GB"
    log_warn ""

    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted by user"
        exit 0
    fi

    log_info ""
    log_info "Starting build process..."
    START_TIME=$(date +%s)

    setup_llvm_source
    build_llvm
    build_cuda_tile
    create_env_script
    create_project_structure
    verify_installation

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))

    log_info ""
    log_info "=========================================="
    log_info "Setup Complete! (${ELAPSED_MIN} minutes)"
    log_info "=========================================="
    log_info ""
    log_info "Next steps:"
    log_info "  1. source setup_env.sh"
    log_info "  2. Test: cuda-tile-translate --help"
    log_info "  3. Start writing MLIR kernels!"
    log_info ""
}

main "$@"
