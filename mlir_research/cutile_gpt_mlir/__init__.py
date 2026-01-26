# SPDX-License-Identifier: Apache-2.0
"""
cutileGPT MLIR - True Tile Philosophy Implementation

This module provides MLIR-based kernels that follow the true Tile IR philosophy:
- Declarative programming
- Compiler-driven optimization
- Hardware abstraction

Compare with cutile_gpt (Python API):
- cutile_gpt: Tile API usage, but PTX-style thinking
- cutile_gpt_mlir: True tile-based thinking, compiler-driven
"""

from .model import CutileGPTMLIR, CutileGPTConfigMLIR
from .kernels import MLIRKernelLoader

__all__ = ["CutileGPTMLIR", "CutileGPTConfigMLIR", "MLIRKernelLoader"]
