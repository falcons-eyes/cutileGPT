# SPDX-License-Identifier: Apache-2.0
"""
cutileGPT - GPT implementation using NVIDIA cuda.tile

A high-performance GPT implementation using Tile Programming philosophy.

Structure:
- kernels/   : Low-level CUDA kernels (cutile)
- api/       : High-level Tile API (declarative builder)
- models/    : GPT model implementations
- utils/     : Utilities (HF loader, benchmark)
- examples/  : Educational Tile Philosophy examples

Quick Start:
    from cutile_gpt.models import CutileGPT, GPTConfig
    from cutile_gpt.api import tile, configure_tiles

    # Create a model
    config = GPTConfig.gpt2()
    model = CutileGPT(config)
    model.load_from_huggingface('gpt2')

    # Use Tile API for custom operations
    result = tile(x, "input").linear(w, b).gelu().execute()
"""

# Models
from .models import CutileGPT, GPTConfig

# High-level Tile API
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

# Low-level kernels
from .kernels import (
    cutile_gelu,
    cutile_embedding,
    cutile_linear,
    cutile_linear_bias,
    cutile_layer_norm,
    cutile_causal_attention,
    cutile_fused_mlp,
)

# Utils
from .utils import (
    HFWeightLoader,
    benchmark_cupy,
    benchmark_torch,
)

__version__ = "0.2.0"

__all__ = [
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
    'HFWeightLoader',
    'benchmark_cupy',
    'benchmark_torch',
]
