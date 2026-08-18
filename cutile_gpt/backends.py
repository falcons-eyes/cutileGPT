# SPDX-License-Identifier: Apache-2.0
"""Composable baseline implementations for semantic region autotuning.

These paths are deliberately built from smaller, already-correct primitives.
They provide a real alternative to each fused kernel, make fusion benefits
measurable, and remain useful fallbacks when a fused candidate rejects a new
shape or dtype.
"""

from __future__ import annotations

import cupy as cp


def separate_qk_norm_rope(
    q, k, q_weight, k_weight, cos, sin, *, eps: float = 1e-6
):
    from .kernels import cutile_rms_norm, cutile_rope_qk

    q_norm = cutile_rms_norm(q, q_weight, eps=eps)
    k_norm = cutile_rms_norm(k, k_weight, eps=eps)
    return cutile_rope_qk(q_norm, k_norm, cos, sin)


def separate_qk_norm_rope_cached(
    q,
    k,
    v,
    q_weight,
    k_weight,
    cos,
    sin,
    k_slot,
    v_slot,
    *,
    eps: float = 1e-6,
):
    q_out, k_out = separate_qk_norm_rope(
        q, k, q_weight, k_weight, cos, sin, eps=eps
    )
    cp.copyto(k_slot, k_out)
    cp.copyto(v_slot, v)
    return q_out


def separate_linear_residual(
    x,
    weight,
    residual,
    bias=None,
    weight_t=None,
):
    from .kernels import cutile_linear, cutile_linear_bias

    if bias is None:
        projected = cutile_linear(x, weight, weight_t)
    else:
        projected = cutile_linear_bias(x, weight, bias, weight_t)
    return projected + residual


def separate_swiglu_residual(
    x,
    w_gate,
    w_up,
    w_down,
    residual=None,
):
    from .kernels import cutile_linear, cutile_linear_residual

    gate = cutile_linear(x, w_gate)
    up = cutile_linear(x, w_up)
    hidden = gate / (1.0 + cp.exp(-gate)) * up
    if residual is None:
        return cutile_linear(hidden, w_down)
    return cutile_linear_residual(hidden, w_down, residual)


def separate_gelu_mlp_residual(
    x,
    w_fc,
    b_fc,
    w_proj,
    b_proj,
    residual=None,
    w_proj_t=None,
):
    from .kernels import (
        cutile_gelu,
        cutile_linear_bias,
        cutile_linear_residual,
    )

    hidden = cutile_gelu(cutile_linear_bias(x, w_fc, b_fc))
    if residual is None:
        return cutile_linear_bias(hidden, w_proj, b_proj, w_proj_t)
    return cutile_linear_residual(
        hidden, w_proj, residual, bias=b_proj, weight_t=w_proj_t
    )
