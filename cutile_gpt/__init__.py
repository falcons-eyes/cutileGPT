# SPDX-License-Identifier: Apache-2.0
"""
cutileGPT - GPT implementation using NVIDIA cuda.tile

A high-performance GPT implementation using Tile Programming philosophy.

Structure:
- kernels/   : Low-level CUDA kernels (cutile) - core, no external deps
- api/       : High-level Tile API (declarative builder)
- models/    : GPT model implementations (requires optional deps)
- utils/     : Utilities (HF loader, benchmark)
- examples/  : Educational Tile Philosophy examples

Installation:
    pip install cutile-gpt              # Core only
    pip install cutile-gpt[hf]          # + HuggingFace support
    pip install cutile-gpt[torch]       # + PyTorch benchmarking
    pip install cutile-gpt[all]         # Everything

Quick Start (Core):
    from cutile_gpt import tile, cutile_gelu, cutile_linear

    # Use Tile API for custom operations
    result = tile(x, "input").linear(w, b).gelu().execute()

    # Or use kernels directly
    y = cutile_gelu(x)

Quick Start (with HuggingFace):
    from cutile_gpt import CutileGPT, GPTConfig

    model = CutileGPT(GPTConfig.gpt2())
    model.load_from_huggingface('gpt2')
"""

__version__ = "0.2.0"

# =============================================================================
# Core: Low-level kernels (always available)
# =============================================================================
from .kernels import (
    cutile_gelu,
    cutile_embedding,
    cutile_linear,
    cutile_linear_bias,
    cutile_layer_norm,
    cutile_causal_attention,
    cutile_fused_mlp,
)

# =============================================================================
# Core: High-level Tile API (always available)
# =============================================================================
from .api import (
    TileOp,
    TileConfig,
    TensorSpec,
    Layout,
    DType,
    tile,
    configure_tiles,
    DataProfile,
    DataAnalyzer,
)

# =============================================================================
# Optional: Models (requires transformers for HF loading)
# =============================================================================
from .models import CutileGPT, GPTConfig

# =============================================================================
# Optional: Utils (some require torch/transformers)
# =============================================================================
from .utils import benchmark_cupy

# Lazy imports for optional dependencies
def __getattr__(name):
    if name == "HFWeightLoader":
        try:
            from .utils import HFWeightLoader
            return HFWeightLoader
        except ImportError:
            raise ImportError(
                "HFWeightLoader requires 'transformers'. "
                "Install with: pip install cutile-gpt[hf]"
            )

    if name == "benchmark_torch":
        try:
            from .utils import benchmark_torch
            return benchmark_torch
        except ImportError:
            raise ImportError(
                "benchmark_torch requires 'torch'. "
                "Install with: pip install cutile-gpt[torch]"
            )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    '__version__',
    # Models
    'CutileGPT',
    'GPTConfig',
    # Tile API
    'TileOp',
    'TileConfig',
    'TensorSpec',
    'Layout',
    'DType',
    'tile',
    'configure_tiles',
    'DataProfile',
    'DataAnalyzer',
    # Kernels
    'cutile_gelu',
    'cutile_embedding',
    'cutile_linear',
    'cutile_linear_bias',
    'cutile_layer_norm',
    'cutile_causal_attention',
    'cutile_fused_mlp',
    # Utils
    'benchmark_cupy',
    'benchmark_torch',  # lazy import
    'HFWeightLoader',   # lazy import
]
