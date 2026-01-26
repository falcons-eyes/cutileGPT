# SPDX-License-Identifier: Apache-2.0
#
# cutileGPT with MLIR support
#
# This CMakeLists.txt provides:
# 1. MLIR kernel compilation infrastructure
# 2. Automated bytecode generation
# 3. Python integration
#

cmake_minimum_required(VERSION 3.20)
project(cutileGPT VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(CUTILEGPT_BUILD_MLIR "Build MLIR kernels" ON)
option(CUTILEGPT_BUILD_TESTS "Build tests" ON)
option(CUTILEGPT_BUILD_PYTHON "Build Python bindings" ON)

# Find CUDA Tile
if(CUTILEGPT_BUILD_MLIR)
    set(CUDA_TILE_DIR "${CMAKE_SOURCE_DIR}/tools/cuda-tile" CACHE PATH "CUDA Tile installation directory")

    if(NOT EXISTS "${CUDA_TILE_DIR}")
        message(WARNING "CUDA Tile not found at ${CUDA_TILE_DIR}")
        message(WARNING "Run ./setup_cuda_tile.sh first")
        set(CUTILEGPT_BUILD_MLIR OFF)
    else()
        message(STATUS "CUDA Tile found: ${CUDA_TILE_DIR}")

        # Add CUDA Tile to path
        set(CMAKE_PREFIX_PATH "${CUDA_TILE_DIR}" ${CMAKE_PREFIX_PATH})

        # Find tools
        find_program(CUDA_TILE_TRANSLATE cuda-tile-translate
            PATHS "${CUDA_TILE_DIR}/bin"
            REQUIRED)

        find_program(MLIR_OPT mlir-opt
            PATHS "${CUDA_TILE_DIR}/../llvm/bin"
            REQUIRED)

        message(STATUS "cuda-tile-translate: ${CUDA_TILE_TRANSLATE}")
        message(STATUS "mlir-opt: ${MLIR_OPT}")
    endif()
endif()

# Find CUDA (optional, for tileiras)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit ${CUDAToolkit_VERSION} found")

    find_program(TILEIRAS tileiras
        PATHS "${CUDAToolkit_BIN_DIR}"
        NO_DEFAULT_PATH)

    if(TILEIRAS)
        message(STATUS "tileiras found: ${TILEIRAS}")
    else()
        message(STATUS "tileiras not found (AoT compilation disabled)")
    endif()
else()
    message(STATUS "CUDA Toolkit not found (JIT mode only)")
endif()

# Kernel compilation function
function(add_mlir_kernel TARGET_NAME)
    set(options AOT)  # Ahead-of-time compile to cubin
    set(oneValueArgs MLIR_FILE GPU_ARCH BYTECODE_VERSION)
    set(multiValueArgs)

    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_MLIR_FILE)
        message(FATAL_ERROR "MLIR_FILE is required")
    endif()

    if(NOT ARG_BYTECODE_VERSION)
        set(ARG_BYTECODE_VERSION "13.1")
    endif()

    get_filename_component(MLIR_NAME ${ARG_MLIR_FILE} NAME_WE)
    set(BYTECODE_FILE "${CMAKE_BINARY_DIR}/kernels/${MLIR_NAME}.tilebc")

    # MLIR -> Bytecode
    add_custom_command(
        OUTPUT ${BYTECODE_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/kernels"
        COMMAND ${CUDA_TILE_TRANSLATE}
            ${ARG_MLIR_FILE}
            --bytecode-version=${ARG_BYTECODE_VERSION}
            --mlir-to-cudatilebc
            --no-implicit-module
            -o ${BYTECODE_FILE}
        DEPENDS ${ARG_MLIR_FILE}
        COMMENT "Compiling MLIR: ${MLIR_NAME}.mlir -> ${MLIR_NAME}.tilebc"
        VERBATIM
    )

    # Bytecode -> Cubin (if AOT requested and tileiras available)
    if(ARG_AOT AND TILEIRAS)
        if(NOT ARG_GPU_ARCH)
            message(WARNING "GPU_ARCH not specified for ${TARGET_NAME}, using sm_100")
            set(ARG_GPU_ARCH "sm_100")
        endif()

        set(CUBIN_FILE "${CMAKE_BINARY_DIR}/kernels/${MLIR_NAME}.cubin")

        add_custom_command(
            OUTPUT ${CUBIN_FILE}
            COMMAND ${TILEIRAS}
                --gpu-name ${ARG_GPU_ARCH}
                ${BYTECODE_FILE}
                -o ${CUBIN_FILE}
            DEPENDS ${BYTECODE_FILE}
            COMMENT "AoT compiling: ${MLIR_NAME}.tilebc -> ${MLIR_NAME}.cubin (${ARG_GPU_ARCH})"
            VERBATIM
        )

        add_custom_target(${TARGET_NAME} ALL DEPENDS ${CUBIN_FILE})
    else()
        add_custom_target(${TARGET_NAME} ALL DEPENDS ${BYTECODE_FILE})
    endif()

    # Install
    install(FILES ${BYTECODE_FILE}
        DESTINATION cutile_gpt_mlir/compiled)

    if(ARG_AOT AND TILEIRAS AND EXISTS ${CUBIN_FILE})
        install(FILES ${CUBIN_FILE}
            DESTINATION cutile_gpt_mlir/compiled)
    endif()
endfunction()

# Add MLIR kernels
if(CUTILEGPT_BUILD_MLIR)
    add_subdirectory(cutile_gpt_mlir/kernels)
endif()

# Python package
if(CUTILEGPT_BUILD_PYTHON)
    # Install Python files
    install(DIRECTORY cutile_gpt/ DESTINATION cutile_gpt
        FILES_MATCHING PATTERN "*.py")

    if(CUTILEGPT_BUILD_MLIR)
        install(DIRECTORY cutile_gpt_mlir/ DESTINATION cutile_gpt_mlir
            FILES_MATCHING PATTERN "*.py")
    endif()
endif()

# Tests
if(CUTILEGPT_BUILD_TESTS)
    enable_testing()
    # add_subdirectory(tests)
endif()

# Summary
message(STATUS "")
message(STATUS "cutileGPT Configuration Summary:")
message(STATUS "  MLIR kernels: ${CUTILEGPT_BUILD_MLIR}")
message(STATUS "  Python: ${CUTILEGPT_BUILD_PYTHON}")
message(STATUS "  Tests: ${CUTILEGPT_BUILD_TESTS}")
if(CUTILEGPT_BUILD_MLIR)
    message(STATUS "  AoT compilation: ${TILEIRAS}")
endif()
message(STATUS "")
