# SPDX-License-Identifier: Apache-2.0
"""Correctness and selection checks for competing semantic region paths."""

import pytest

cp = pytest.importorskip("cupy")

from cutile_gpt.backends import (
    separate_linear_residual,
    separate_qk_norm_rope,
    separate_swiglu_residual,
)
from cutile_gpt.kernels import (
    cutile_linear_residual,
    cutile_qk_norm_rope,
    cutile_swiglu_mlp,
    rope_tables,
)
from cutile_gpt.planner import (
    Backend,
    ExecutionPhase,
    KernelCandidate,
    KernelRegistry,
    RegionKind,
    TacticCache,
)
from cutile_gpt.regions import TileRuntime


def gpu_available():
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not gpu_available(), reason="needs a CUDA GPU")


def test_separate_qk_region_matches_fused_candidate():
    cp.random.seed(1)
    q = cp.random.standard_normal((1, 8, 9, 64), dtype=cp.float32)
    k = cp.random.standard_normal((1, 2, 9, 64), dtype=cp.float32)
    qw = cp.random.random(64, dtype=cp.float32) + 0.5
    kw = cp.random.random(64, dtype=cp.float32) + 0.5
    cos, sin = rope_tables(9, 64)

    fused = cutile_qk_norm_rope(q, k, qw, kw, cos, sin)
    separate = separate_qk_norm_rope(q, k, qw, kw, cos, sin)
    cp.cuda.get_current_stream().synchronize()

    for expected, actual in zip(fused, separate, strict=True):
        assert float(cp.max(cp.abs(expected - actual))) < 1e-5


def test_separate_linear_residual_matches_fused_candidate():
    cp.random.seed(2)
    x = cp.random.standard_normal((7, 64), dtype=cp.float32) * 0.1
    weight = cp.random.standard_normal((96, 64), dtype=cp.float32) * 0.02
    bias = cp.random.standard_normal(96, dtype=cp.float32) * 0.01
    residual = cp.random.standard_normal((7, 96), dtype=cp.float32) * 0.1

    fused = cutile_linear_residual(x, weight, residual, bias)
    separate = separate_linear_residual(x, weight, residual, bias)
    cp.cuda.get_current_stream().synchronize()

    assert float(cp.max(cp.abs(fused - separate))) < 1e-5


def test_separate_swiglu_region_matches_fused_candidate():
    cp.random.seed(3)
    x = cp.random.standard_normal((1, 8, 64), dtype=cp.float32) * 0.1
    gate = cp.random.standard_normal((128, 64), dtype=cp.float32) * 0.02
    up = cp.random.standard_normal((128, 64), dtype=cp.float32) * 0.02
    down = cp.random.standard_normal((64, 128), dtype=cp.float32) * 0.02
    residual = cp.random.standard_normal((1, 8, 64), dtype=cp.float32) * 0.1

    fused = cutile_swiglu_mlp(x, gate, up, down, residual=residual)
    separate = separate_swiglu_residual(
        x, gate, up, down, residual=residual
    )
    cp.cuda.get_current_stream().synchronize()

    assert float(cp.max(cp.abs(fused - separate))) < 1e-5


def test_runtime_autotune_validates_and_caches_candidates():
    cache = TacticCache()
    registry = KernelRegistry(cache)
    registry.register(
        RegionKind.ATTENTION,
        KernelCandidate("view", Backend.CUTILE, lambda x: x, priority=1),
    )
    registry.register(
        RegionKind.ATTENTION,
        KernelCandidate("copy", Backend.CUTILE, lambda x: x.copy()),
    )
    runtime = TileRuntime(
        registry, autotune=True, tuning_warmup=1, tuning_iterations=2
    )
    x = cp.arange(64, dtype=cp.float32)

    output = runtime.run(
        RegionKind.ATTENTION,
        x,
        phase=ExecutionPhase.DECODE,
        site="test",
        shape_bucket=1,
    )
    cp.cuda.get_current_stream().synchronize()

    assert bool(cp.all(output == x))
    assert len(runtime.tuning_results) == 1
    assert all(item.valid for item in runtime.tuning_results[0].measurements)
    selected = cache.get(runtime.tuning_results[0].region)
    assert selected is not None
    assert "latency_ms" in selected.parameters
