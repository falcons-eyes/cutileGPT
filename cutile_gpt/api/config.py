# SPDX-License-Identifier: Apache-2.0
"""
Tile API Configuration Classes

Data structures for explicit, declarative tile programming.
"""

from dataclasses import dataclass
from typing import Tuple
from enum import Enum, auto


class Layout(Enum):
    """Memory layout for tensors."""
    ROW_MAJOR = auto()      # C-contiguous (default)
    COL_MAJOR = auto()      # Fortran-contiguous
    TILED = auto()          # Blocked/tiled layout


class DType(Enum):
    """Supported data types."""
    FLOAT32 = auto()
    FLOAT16 = auto()
    BFLOAT16 = auto()

    def to_cupy(self):
        import cupy as cp
        mapping = {
            DType.FLOAT32: cp.float32,
            DType.FLOAT16: cp.float16,
            DType.BFLOAT16: cp.dtype('bfloat16'),
        }
        return mapping[self]


@dataclass
class TileConfig:
    """
    Explicit tile configuration for operations.

    Tile Programming Philosophy: Make tile sizes explicit and configurable,
    rather than hiding them inside kernels.
    """
    tile_m: int = 64
    tile_n: int = 64
    tile_k: int = 32

    # Hardware hints
    num_ctas: int = 1
    occupancy: int = 4
    latency_hint: int = 4

    # Features
    use_tma: bool = True          # Tensor Memory Accelerator
    use_swizzle: bool = True      # L2 cache optimization
    use_tf32: bool = True         # TF32 for float32 inputs

    def __post_init__(self):
        # Validate power of 2
        for name, val in [('tile_m', self.tile_m),
                          ('tile_n', self.tile_n),
                          ('tile_k', self.tile_k)]:
            if val & (val - 1) != 0:
                raise ValueError(f"{name}={val} must be power of 2")


@dataclass
class TensorSpec:
    """
    Explicit tensor specification.

    Tile Programming Philosophy: Know your data's shape, layout, and type
    at every step of the computation.
    """
    shape: Tuple[int, ...]
    dtype: DType = DType.FLOAT32
    layout: Layout = Layout.ROW_MAJOR
    name: str = ""
    dim_names: Tuple[str, ...] = ()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        result = 1
        for s in self.shape:
            result *= s
        return result

    def with_shape(self, *shape) -> 'TensorSpec':
        return TensorSpec(
            shape=shape,
            dtype=self.dtype,
            layout=self.layout,
            name=self.name,
            dim_names=self.dim_names
        )

    def __repr__(self) -> str:
        dims = ""
        if self.dim_names:
            dims = f", dims={self.dim_names}"
        return f"TensorSpec({self.name}: {self.shape}, {self.dtype.name}, {self.layout.name}{dims})"


def configure_tiles(
    tile_m: int = 64,
    tile_n: int = 64,
    tile_k: int = 32,
    **kwargs
) -> TileConfig:
    """
    Create a tile configuration.

    Example:
        >>> config = configure_tiles(tile_m=128, tile_n=128, use_tma=True)
    """
    return TileConfig(tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, **kwargs)
