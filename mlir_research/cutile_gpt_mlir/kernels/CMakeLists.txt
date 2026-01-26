# SPDX-License-Identifier: Apache-2.0
#
# MLIR Kernels for cutileGPT
#

# Get GPU architecture
if(NOT GPU_ARCH)
    # Try to detect from nvidia-smi
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE GPU_CAP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(GPU_CAP)
        string(REPLACE "." "" GPU_CAP_NO_DOT "${GPU_CAP}")
        set(GPU_ARCH "sm_${GPU_CAP_NO_DOT}")
        message(STATUS "Detected GPU architecture: ${GPU_ARCH}")
    else()
        set(GPU_ARCH "sm_100")  # Blackwell default
        message(STATUS "GPU architecture not detected, using: ${GPU_ARCH}")
    endif()
endif()

# LayerNorm kernel
add_mlir_kernel(layernorm_kernel
    MLIR_FILE ${CMAKE_CURRENT_SOURCE_DIR}/layernorm.mlir
    GPU_ARCH ${GPU_ARCH}
    AOT  # Compile to cubin
)

# TODO: Add more kernels
# add_mlir_kernel(linear_kernel ...)
# add_mlir_kernel(attention_kernel ...)
