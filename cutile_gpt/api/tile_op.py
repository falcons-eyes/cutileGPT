# SPDX-License-Identifier: Apache-2.0
"""
TileOp - Fluent Builder API for Tile Programming

This module provides a builder-style API that makes data flow explicit,
following the Tile Programming philosophy of "declare WHAT, not HOW".

Example:
    result = (
        tile(x, name="input")
            .declare_shape(batch=2, seq=128, dim=768)
            .linear(weight, bias, out_features=3072)
            .gelu()
            .execute()
    )
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum, auto
import math
import cupy as cp

from .config import TileConfig, TensorSpec, Layout, DType


class OpType(Enum):
    """Types of operations in the computation graph."""
    INPUT = auto()
    LINEAR = auto()
    LINEAR_BIAS = auto()
    GELU = auto()
    LAYERNORM = auto()
    ATTENTION = auto()
    FUSED_MLP = auto()
    RESHAPE = auto()
    TRANSPOSE = auto()


@dataclass
class OpNode:
    """A node in the computation graph."""
    op_type: OpType
    inputs: List[Any] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    output_spec: Optional[TensorSpec] = None
    tile_config: Optional[TileConfig] = None

    def __repr__(self) -> str:
        return f"OpNode({self.op_type.name}, out={self.output_spec})"


class TileOp:
    """
    Fluent Builder API for Tile Programming.

    This class implements the Builder pattern with method chaining,
    making the data flow explicit and declarative.
    """

    def __init__(
        self,
        data: Optional[cp.ndarray] = None,
        spec: Optional[TensorSpec] = None,
        ops: Optional[List[OpNode]] = None,
        tile_config: Optional[TileConfig] = None
    ):
        self._data = data
        self._spec = spec
        self._ops: List[OpNode] = ops or []
        self._tile_config = tile_config or TileConfig()
        self._name = spec.name if spec else "tensor"

    @classmethod
    def from_tensor(
        cls,
        tensor: cp.ndarray,
        name: str = "input",
        layout: Layout = Layout.ROW_MAJOR
    ) -> TileOp:
        """Create a TileOp from an existing CuPy tensor."""
        if not isinstance(tensor, cp.ndarray):
            raise ValueError(f"Expected CuPy array, got {type(tensor)}")

        dtype_map = {
            cp.float32: DType.FLOAT32,
            cp.float16: DType.FLOAT16,
        }
        dtype = dtype_map.get(tensor.dtype.type, DType.FLOAT32)

        spec = TensorSpec(
            shape=tensor.shape,
            dtype=dtype,
            layout=layout,
            name=name
        )

        op = OpNode(
            op_type=OpType.INPUT,
            inputs=[tensor],
            output_spec=spec
        )

        return cls(data=tensor, spec=spec, ops=[op])

    @classmethod
    def placeholder(
        cls,
        shape: tuple,
        dtype: DType = DType.FLOAT32,
        name: str = "placeholder"
    ) -> TileOp:
        """Create a placeholder TileOp (for graph building without data)."""
        spec = TensorSpec(shape=shape, dtype=dtype, name=name)
        op = OpNode(op_type=OpType.INPUT, output_spec=spec)
        return cls(spec=spec, ops=[op])

    def declare_shape(self, **dims) -> TileOp:
        """Declare semantic dimension names for the tensor."""
        dim_names = tuple(dims.keys())
        expected_shape = tuple(dims.values())

        if self._spec and self._spec.shape != expected_shape:
            raise ValueError(
                f"Declared shape {expected_shape} doesn't match "
                f"actual shape {self._spec.shape}"
            )

        if self._spec:
            self._spec = TensorSpec(
                shape=self._spec.shape,
                dtype=self._spec.dtype,
                layout=self._spec.layout,
                name=self._spec.name,
                dim_names=dim_names
            )

        return self

    def with_tile_config(self, config: TileConfig) -> TileOp:
        """Set tile configuration for subsequent operations."""
        self._tile_config = config
        return self

    def linear(
        self,
        weight: cp.ndarray,
        bias: Optional[cp.ndarray] = None,
        out_features: Optional[int] = None,
        weight_transposed: bool = False
    ) -> TileOp:
        """Declare a linear transformation: y = x @ W^T + b"""
        in_features = self._spec.shape[-1]

        if weight_transposed:
            actual_out = weight.shape[1]
            actual_in = weight.shape[0]
        else:
            actual_out = weight.shape[0]
            actual_in = weight.shape[1]

        if actual_in != in_features:
            raise ValueError(
                f"Weight in_features ({actual_in}) doesn't match "
                f"input dim ({in_features})"
            )

        if out_features and out_features != actual_out:
            raise ValueError(
                f"Declared out_features ({out_features}) doesn't match "
                f"weight shape ({actual_out})"
            )

        out_shape = self._spec.shape[:-1] + (actual_out,)
        out_spec = TensorSpec(
            shape=out_shape,
            dtype=self._spec.dtype,
            layout=self._spec.layout,
            name=f"{self._name}_linear",
            dim_names=self._spec.dim_names[:-1] + ("out_features",) if self._spec.dim_names else ()
        )

        op_type = OpType.LINEAR_BIAS if bias is not None else OpType.LINEAR
        op = OpNode(
            op_type=op_type,
            inputs=[weight, bias] if bias is not None else [weight],
            params={
                "in_features": in_features,
                "out_features": actual_out,
                "weight_transposed": weight_transposed,
            },
            output_spec=out_spec,
            tile_config=self._tile_config
        )

        return TileOp(
            spec=out_spec,
            ops=self._ops + [op],
            tile_config=self._tile_config
        )

    def gelu(self, approximate: bool = True) -> TileOp:
        """Declare GELU activation."""
        out_spec = TensorSpec(
            shape=self._spec.shape,
            dtype=self._spec.dtype,
            layout=self._spec.layout,
            name=f"{self._name}_gelu",
            dim_names=self._spec.dim_names
        )

        op = OpNode(
            op_type=OpType.GELU,
            params={"approximate": approximate},
            output_spec=out_spec,
            tile_config=self._tile_config
        )

        return TileOp(
            spec=out_spec,
            ops=self._ops + [op],
            tile_config=self._tile_config
        )

    def layernorm(
        self,
        weight: cp.ndarray,
        bias: cp.ndarray,
        eps: float = 1e-5,
        normalized_shape: Optional[tuple] = None
    ) -> TileOp:
        """Declare Layer Normalization."""
        norm_shape = normalized_shape or (self._spec.shape[-1],)

        if weight.shape != norm_shape:
            raise ValueError(
                f"Weight shape {weight.shape} doesn't match "
                f"normalized_shape {norm_shape}"
            )

        out_spec = TensorSpec(
            shape=self._spec.shape,
            dtype=self._spec.dtype,
            layout=self._spec.layout,
            name=f"{self._name}_ln",
            dim_names=self._spec.dim_names
        )

        op = OpNode(
            op_type=OpType.LAYERNORM,
            inputs=[weight, bias],
            params={"eps": eps, "normalized_shape": norm_shape},
            output_spec=out_spec,
            tile_config=self._tile_config
        )

        return TileOp(
            spec=out_spec,
            ops=self._ops + [op],
            tile_config=self._tile_config
        )

    def attention(
        self,
        n_head: int,
        head_dim: Optional[int] = None,
        causal: bool = True,
        scale: Optional[float] = None
    ) -> TileOp:
        """Declare self-attention operation."""
        n_embd = self._spec.shape[-1]

        if head_dim is None:
            if n_embd % 3 == 0:
                actual_embd = n_embd // 3
                head_dim = actual_embd // n_head
            else:
                head_dim = n_embd // n_head

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        if n_embd % 3 == 0:
            out_embd = n_embd // 3
        else:
            out_embd = n_embd

        out_shape = self._spec.shape[:-1] + (out_embd,)

        out_spec = TensorSpec(
            shape=out_shape,
            dtype=self._spec.dtype,
            layout=self._spec.layout,
            name=f"{self._name}_attn",
            dim_names=self._spec.dim_names
        )

        op = OpNode(
            op_type=OpType.ATTENTION,
            params={
                "n_head": n_head,
                "head_dim": head_dim,
                "causal": causal,
                "scale": scale,
            },
            output_spec=out_spec,
            tile_config=self._tile_config
        )

        return TileOp(
            spec=out_spec,
            ops=self._ops + [op],
            tile_config=self._tile_config
        )

    def fused_mlp(
        self,
        w_fc: cp.ndarray,
        b_fc: cp.ndarray,
        w_proj: cp.ndarray,
        b_proj: cp.ndarray,
        hidden_features: Optional[int] = None
    ) -> TileOp:
        """Declare fused MLP: Linear -> GELU -> Linear"""
        in_features = self._spec.shape[-1]
        actual_hidden = w_fc.shape[0]
        out_features = w_proj.shape[0]

        if hidden_features and hidden_features != actual_hidden:
            raise ValueError(
                f"Declared hidden_features ({hidden_features}) doesn't match "
                f"w_fc shape ({actual_hidden})"
            )

        out_shape = self._spec.shape[:-1] + (out_features,)

        out_spec = TensorSpec(
            shape=out_shape,
            dtype=self._spec.dtype,
            layout=self._spec.layout,
            name=f"{self._name}_mlp",
            dim_names=self._spec.dim_names
        )

        op = OpNode(
            op_type=OpType.FUSED_MLP,
            inputs=[w_fc, b_fc, w_proj, b_proj],
            params={
                "in_features": in_features,
                "hidden_features": actual_hidden,
                "out_features": out_features,
            },
            output_spec=out_spec,
            tile_config=self._tile_config
        )

        return TileOp(
            spec=out_spec,
            ops=self._ops + [op],
            tile_config=self._tile_config
        )

    def reshape(self, *shape) -> TileOp:
        """Declare a reshape operation."""
        new_shape = list(shape)
        neg_idx = None
        known_size = 1

        for i, s in enumerate(new_shape):
            if s == -1:
                if neg_idx is not None:
                    raise ValueError("Only one dimension can be -1")
                neg_idx = i
            else:
                known_size *= s

        if neg_idx is not None:
            new_shape[neg_idx] = self._spec.numel // known_size

        new_shape = tuple(new_shape)

        out_spec = TensorSpec(
            shape=new_shape,
            dtype=self._spec.dtype,
            layout=self._spec.layout,
            name=f"{self._name}_reshape"
        )

        op = OpNode(
            op_type=OpType.RESHAPE,
            params={"shape": new_shape},
            output_spec=out_spec
        )

        return TileOp(
            spec=out_spec,
            ops=self._ops + [op],
            tile_config=self._tile_config
        )

    def transpose(self, *dims) -> TileOp:
        """Declare a transpose operation."""
        if not dims:
            dims = tuple(range(self._spec.ndim - 2)) + (self._spec.ndim - 1, self._spec.ndim - 2)

        new_shape = tuple(self._spec.shape[d] for d in dims)
        new_dim_names = tuple(self._spec.dim_names[d] for d in dims) if self._spec.dim_names else ()

        out_spec = TensorSpec(
            shape=new_shape,
            dtype=self._spec.dtype,
            layout=self._spec.layout,
            name=f"{self._name}_transpose",
            dim_names=new_dim_names
        )

        op = OpNode(
            op_type=OpType.TRANSPOSE,
            params={"dims": dims},
            output_spec=out_spec
        )

        return TileOp(
            spec=out_spec,
            ops=self._ops + [op],
            tile_config=self._tile_config
        )

    def execute(self, optimize: bool = True) -> cp.ndarray:
        """Execute the declared computation graph."""
        if optimize:
            self._optimize_graph()
        return self._execute_ops()

    def _optimize_graph(self):
        """Optimize the computation graph (placeholder for future optimizations)."""
        pass

    def _execute_ops(self) -> cp.ndarray:
        """Execute the operation graph."""
        from ..kernels import (
            cutile_linear, cutile_linear_bias, cutile_gelu,
            cutile_layer_norm, cutile_causal_attention, cutile_fused_mlp
        )

        current = self._data

        for op in self._ops:
            if op.op_type == OpType.INPUT:
                if op.inputs:
                    current = op.inputs[0]
                continue

            elif op.op_type == OpType.LINEAR:
                weight = op.inputs[0]
                current = cutile_linear(current, weight)

            elif op.op_type == OpType.LINEAR_BIAS:
                weight, bias = op.inputs
                current = cutile_linear_bias(current, weight, bias)

            elif op.op_type == OpType.GELU:
                current = cutile_gelu(current)

            elif op.op_type == OpType.LAYERNORM:
                weight, bias = op.inputs
                eps = op.params.get("eps", 1e-5)
                current = cutile_layer_norm(current, weight, bias, eps)

            elif op.op_type == OpType.ATTENTION:
                n_head = op.params["n_head"]
                batch, seq, three_embd = current.shape
                n_embd = three_embd // 3
                head_dim = n_embd // n_head

                q, k, v = cp.split(current, 3, axis=2)
                q = cp.transpose(cp.reshape(q, (batch, seq, n_head, head_dim)), (0, 2, 1, 3))
                k = cp.transpose(cp.reshape(k, (batch, seq, n_head, head_dim)), (0, 2, 1, 3))
                v = cp.transpose(cp.reshape(v, (batch, seq, n_head, head_dim)), (0, 2, 1, 3))

                y = cutile_causal_attention(q, k, v, n_head)

                y = cp.transpose(y, (0, 2, 1, 3))
                if not y.flags.c_contiguous:
                    y = cp.ascontiguousarray(y)
                current = cp.reshape(y, (batch, seq, n_embd))

            elif op.op_type == OpType.FUSED_MLP:
                w_fc, b_fc, w_proj, b_proj = op.inputs
                current = cutile_fused_mlp(current, w_fc, b_fc, w_proj, b_proj)

            elif op.op_type == OpType.RESHAPE:
                current = cp.reshape(current, op.params["shape"])

            elif op.op_type == OpType.TRANSPOSE:
                current = cp.transpose(current, op.params["dims"])

        return current

    @property
    def spec(self) -> TensorSpec:
        return self._spec

    @property
    def shape(self) -> tuple:
        return self._spec.shape if self._spec else ()

    def describe(self) -> str:
        """Get a human-readable description of the computation graph."""
        lines = ["TileOp Computation Graph:", "=" * 40]

        for i, op in enumerate(self._ops):
            arrow = "→" if i > 0 else " "
            lines.append(f"{arrow} [{i}] {op.op_type.name}")
            if op.output_spec:
                lines.append(f"      Output: {op.output_spec.shape}")
            if op.params:
                for k, v in op.params.items():
                    lines.append(f"      {k}: {v}")

        if self._spec:
            lines.append("=" * 40)
            lines.append(f"Final: {self._spec}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"TileOp({self._spec})"


def tile(tensor: cp.ndarray, name: str = "input") -> TileOp:
    """Shorthand for TileOp.from_tensor()."""
    return TileOp.from_tensor(tensor, name=name)
