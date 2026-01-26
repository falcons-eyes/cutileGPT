# SPDX-License-Identifier: Apache-2.0
"""
MLIR Kernel Loader

Loads and manages MLIR-compiled kernels (bytecode or cubin files).
Provides a clean Python interface to MLIR kernels.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import cupy as cp
from cupy.cuda import driver


class MLIRKernel:
    """
    Wrapper for a single MLIR kernel.

    Handles loading from bytecode/cubin and provides launch interface.
    """

    def __init__(self, module_path: str, kernel_name: str):
        """
        Args:
            module_path: Path to .tilebc or .cubin file
            kernel_name: Name of the kernel function
        """
        self.module_path = module_path
        self.kernel_name = kernel_name

        # Load module
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Kernel file not found: {module_path}")

        self.module = driver.moduleLoad(module_path)
        self.function = self.module.get_function(kernel_name)

        print(f"✓ Loaded MLIR kernel: {kernel_name} from {Path(module_path).name}")

    def launch(
        self,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        args: list,
        shared_mem: int = 0,
        stream: Optional[cp.cuda.Stream] = None,
    ):
        """
        Launch the MLIR kernel.

        Args:
            grid: Grid dimensions (x, y, z)
            block: Block dimensions (x, y, z) - usually (1,1,1) for Tile IR
            args: Kernel arguments (device pointers and scalars)
            shared_mem: Shared memory bytes (usually 0 for Tile IR)
            stream: CUDA stream (default: null stream)
        """
        if stream is None:
            stream = cp.cuda.Stream.null

        # CUDA Driver API launch
        driver.launchKernel(
            self.function,
            *grid,  # gridDimX, gridDimY, gridDimZ
            *block,  # blockDimX, blockDimY, blockDimZ
            shared_mem,
            stream.ptr,
            args,
            None,  # extra parameters
        )

    def __del__(self):
        """Unload module on destruction."""
        if hasattr(self, "module"):
            try:
                driver.moduleUnload(self.module)
            except:
                pass


class MLIRKernelLoader:
    """
    Manages a collection of MLIR kernels.

    Automatically finds compiled kernels and provides easy access.
    """

    def __init__(self, compiled_dir: Optional[str] = None):
        """
        Args:
            compiled_dir: Directory containing compiled kernels
                          If None, uses default: cutile_gpt_mlir/compiled
        """
        if compiled_dir is None:
            # Find compiled directory relative to this file
            this_dir = Path(__file__).parent
            compiled_dir = this_dir / "compiled"

        self.compiled_dir = Path(compiled_dir)

        if not self.compiled_dir.exists():
            raise FileNotFoundError(
                f"Compiled kernels directory not found: {self.compiled_dir}\n"
                f"Run: cmake --build build && cmake --install build"
            )

        self.kernels: Dict[str, MLIRKernel] = {}

        print(f"MLIR Kernel Loader initialized: {self.compiled_dir}")

    def load_kernel(self, kernel_id: str, kernel_file: str, kernel_name: str) -> MLIRKernel:
        """
        Load a specific kernel.

        Args:
            kernel_id: Identifier for this kernel (e.g., "layernorm")
            kernel_file: Filename (e.g., "layernorm.cubin" or "layernorm.tilebc")
            kernel_name: Kernel function name (e.g., "layernorm_kernel")

        Returns:
            MLIRKernel object
        """
        kernel_path = self.compiled_dir / kernel_file

        # Prefer cubin over tilebc (AoT compiled is faster to load)
        if not kernel_path.exists():
            # Try alternative extension
            if kernel_file.endswith(".cubin"):
                alt_path = kernel_path.with_suffix(".tilebc")
            else:
                alt_path = kernel_path.with_suffix(".cubin")

            if alt_path.exists():
                kernel_path = alt_path
                print(f"Note: Using {alt_path.name} (AoT version not found)")
            else:
                raise FileNotFoundError(f"Kernel not found: {kernel_path} or {alt_path}")

        kernel = MLIRKernel(str(kernel_path), kernel_name)
        self.kernels[kernel_id] = kernel
        return kernel

    def get_kernel(self, kernel_id: str) -> MLIRKernel:
        """Get a loaded kernel by ID."""
        if kernel_id not in self.kernels:
            raise KeyError(f"Kernel not loaded: {kernel_id}")
        return self.kernels[kernel_id]

    def load_all_standard_kernels(self):
        """Load all standard cutileGPT kernels."""
        print("Loading standard MLIR kernels...")

        try:
            self.load_kernel("layernorm", "layernorm.cubin", "layernorm_kernel")
        except FileNotFoundError as e:
            print(f"Warning: {e}")

        # TODO: Add more kernels as they are implemented
        # self.load_kernel("linear", "linear.cubin", "matmul_kernel")
        # self.load_kernel("attention", "attention.cubin", "causal_attention_kernel")

        print(f"✓ Loaded {len(self.kernels)} MLIR kernels")

    def __repr__(self):
        return f"MLIRKernelLoader({len(self.kernels)} kernels loaded)"


# Example usage
if __name__ == "__main__":
    print("=== MLIR Kernel Loader Test ===\n")

    try:
        loader = MLIRKernelLoader()
        loader.load_all_standard_kernels()

        print(f"\n{loader}")
        print("\nAvailable kernels:")
        for kernel_id in loader.kernels:
            print(f"  - {kernel_id}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("  1. Run: ./setup_cuda_tile.sh")
        print("  2. Build: cmake --build build")
        print("  3. Install: cmake --install build")
