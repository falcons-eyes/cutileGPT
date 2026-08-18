# SPDX-License-Identifier: Apache-2.0
"""
cutile GPT Kernels

Low-level CUDA kernels using NVIDIA cuda.tile (cutile).
These are optimized GPU kernels for transformer operations.
"""

from .attention import cutile_causal_attention
from .embedding import cutile_embedding
from .fused_mlp import cutile_fused_mlp
from .gelu import cutile_gelu
from .kv_cache import KVCache
from .layernorm import cutile_layer_norm
from .linear import (
    LinearTileConfig,
    autotune_linear,
    cutile_linear,
    cutile_linear_bias,
    cutile_linear_residual,
)
from .rmsnorm import cutile_rms_norm
from .rope import (
    cutile_qk_norm_rope,
    cutile_qk_norm_rope_cached,
    cutile_rope,
    cutile_rope_qk,
    rope_tables,
)
from .swiglu import cutile_swiglu_mlp

__all__ = [
    'KVCache',
    'LinearTileConfig',
    'autotune_linear',
    'cutile_gelu',
    'cutile_embedding',
    'cutile_linear',
    'cutile_linear_bias',
    'cutile_linear_residual',
    'cutile_layer_norm',
    'cutile_causal_attention',
    'cutile_fused_mlp',
    'cutile_rms_norm',
    'cutile_rope',
    'cutile_rope_qk',
    'cutile_qk_norm_rope',
    'cutile_qk_norm_rope_cached',
    'rope_tables',
    'cutile_swiglu_mlp',
]
