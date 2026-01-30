# SPDX-License-Identifier: Apache-2.0
"""
Utilities for cutile GPT

Core utilities (always available):
- benchmark_cupy: Benchmark CuPy functions

Optional utilities:
- HFWeightLoader: Load HuggingFace weights (requires transformers)
- benchmark_torch: Benchmark PyTorch functions (requires torch)
"""

# Core utilities (always available)
from .benchmark import benchmark_cupy, print_benchmark_result, compare_benchmarks

# Optional: benchmark_torch (requires torch)
try:
    from .benchmark import benchmark_torch
except ImportError:
    benchmark_torch = None

# Optional: HFWeightLoader (requires transformers)
try:
    from .hf_loader import HFWeightLoader
except ImportError:
    HFWeightLoader = None

__all__ = [
    'benchmark_cupy',
    'print_benchmark_result',
    'compare_benchmarks',
    'benchmark_torch',
    'HFWeightLoader',
]
