# SPDX-License-Identifier: Apache-2.0
"""
Tile API - High-level Declarative Interface

This module provides a fluent builder API for tile programming,
following the philosophy: "declare WHAT, not HOW".

Example:
    from cutile_gpt.api import tile, configure_tiles

    result = (
        tile(x, "hidden_states")
            .declare_shape(batch=2, seq=128, dim=768)
            .linear(weight, bias, out_features=3072)
            .gelu()
            .execute()
    )
"""

from .config import (
    DType,
    Layout,
    TensorSpec,
    TileConfig,
    configure_tiles,
)
from .profiler import (
    DataAnalyzer,
    DataProfile,
)
from .tile_op import (
    OpNode,
    OpType,
    TileOp,
    tile,
)

__all__ = [
    # Config
    'TileConfig',
    'TensorSpec',
    'Layout',
    'DType',
    'configure_tiles',
    # TileOp
    'TileOp',
    'OpType',
    'OpNode',
    'tile',
    # Profiler
    'DataProfile',
    'DataAnalyzer',
]
