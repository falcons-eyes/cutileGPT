# SPDX-License-Identifier: Apache-2.0
"""Runtime bridge from semantic model regions to tile kernel backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .planner import (
    Backend,
    ExecutionPhase,
    KernelCandidate,
    KernelRegistry,
    RegionCost,
    RegionKind,
    TensorContract,
    TileRegion,
    cuda_target,
    length_bucket,
)


@dataclass(frozen=True)
class PlannedRegion:
    """A lightweight record of what one model step selected."""

    region: TileRegion
    candidate: str
    backend: Backend


@dataclass(frozen=True)
class CandidateMeasurement:
    candidate: str
    backend: Backend
    latency_ms: float | None
    valid: bool
    max_abs_error: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class TuningResult:
    region: TileRegion
    winner: str
    measurements: tuple[CandidateMeasurement, ...]


def _dtype_bytes(dtype: str) -> int:
    if "64" in dtype:
        return 8
    if "16" in dtype:
        return 2
    if "8" in dtype:
        return 1
    return 4


def _tensor_bytes(region: TileRegion, name: str) -> int:
    for spec in region.inputs:
        if spec.name == name:
            return math.prod(spec.shape) * _dtype_bytes(spec.dtype)
    return 0


def _hidden_bytes(region: TileRegion) -> int:
    specs = {spec.name: spec for spec in region.inputs}
    x = specs.get("arg0")
    weight = specs.get("arg1")
    if x is None or weight is None or len(weight.shape) != 2:
        return 0
    rows = math.prod(x.shape[:-1])
    return rows * weight.shape[0] * _dtype_bytes(x.dtype)


def default_kernel_registry() -> KernelRegistry:
    """Register the currently production-ready semantic cuTile regions."""
    from .backends import (
        separate_gelu_mlp_residual,
        separate_linear_residual,
        separate_qk_norm_rope,
        separate_qk_norm_rope_cached,
        separate_swiglu_residual,
    )
    from .kernels import (
        cutile_causal_attention,
        cutile_fused_mlp,
        cutile_linear_residual,
        cutile_qk_norm_rope,
        cutile_qk_norm_rope_cached,
        cutile_swiglu_mlp,
    )

    registry = KernelRegistry()
    def register(kind, name, run, *, priority, cost):
        registry.register(
            kind,
            KernelCandidate(
                name=name,
                backend=Backend.CUTILE,
                run=run,
                priority=priority,
                cost=cost,
            ),
        )

    register(
        RegionKind.QK_NORM_ROPE,
        "cutile.qk_norm_rope",
        cutile_qk_norm_rope,
        priority=10,
        cost=RegionCost(launches=1),
    )
    register(
        RegionKind.QK_NORM_ROPE,
        "cutile.separate_qk_norm_rope",
        separate_qk_norm_rope,
        priority=0,
        cost=lambda region: RegionCost(
            launches=3,
            materialized_bytes=_tensor_bytes(region, "arg0")
            + _tensor_bytes(region, "arg1"),
        ),
    )
    register(
        RegionKind.QK_NORM_ROPE_CACHE,
        "cutile.qk_norm_rope_cache",
        cutile_qk_norm_rope_cached,
        priority=10,
        cost=RegionCost(launches=1),
    )
    register(
        RegionKind.QK_NORM_ROPE_CACHE,
        "cutile.separate_qk_norm_rope_cache",
        separate_qk_norm_rope_cached,
        priority=0,
        cost=lambda region: RegionCost(
            launches=5,
            materialized_bytes=2 * _tensor_bytes(region, "arg0")
            + 2 * _tensor_bytes(region, "arg1")
            + _tensor_bytes(region, "arg2"),
        ),
    )
    register(
        RegionKind.ATTENTION,
        "cutile.gqa_attention",
        cutile_causal_attention,
        priority=10,
        cost=RegionCost(launches=1),
    )
    register(
        RegionKind.LINEAR_RESIDUAL,
        "cutile.linear_residual",
        cutile_linear_residual,
        priority=10,
        cost=RegionCost(launches=1),
    )
    register(
        RegionKind.LINEAR_RESIDUAL,
        "cutile.separate_linear_residual",
        separate_linear_residual,
        priority=0,
        cost=lambda region: RegionCost(
            launches=2,
            materialized_bytes=_tensor_bytes(region, "arg2"),
        ),
    )
    register(
        RegionKind.SWIGLU_RESIDUAL,
        "cutile.swiglu_residual",
        cutile_swiglu_mlp,
        priority=10,
        cost=lambda region: RegionCost(
            launches=2, materialized_bytes=_hidden_bytes(region)
        ),
    )
    register(
        RegionKind.SWIGLU_RESIDUAL,
        "cutile.separate_swiglu_residual",
        separate_swiglu_residual,
        priority=0,
        cost=lambda region: RegionCost(
            launches=4, materialized_bytes=3 * _hidden_bytes(region)
        ),
    )
    register(
        RegionKind.GELU_MLP_RESIDUAL,
        "cutile.gelu_mlp_residual",
        cutile_fused_mlp,
        priority=10,
        cost=lambda region: RegionCost(
            launches=2, materialized_bytes=_hidden_bytes(region)
        ),
    )
    register(
        RegionKind.GELU_MLP_RESIDUAL,
        "cutile.separate_gelu_mlp_residual",
        separate_gelu_mlp_residual,
        priority=0,
        cost=lambda region: RegionCost(
            launches=3, materialized_bytes=2 * _hidden_bytes(region)
        ),
    )
    return registry


class TileRuntime:
    """Execute and expose the semantic tile plan selected for a model step."""

    def __init__(
        self,
        registry: KernelRegistry | None = None,
        *,
        autotune: bool = False,
        tuning_warmup: int = 2,
        tuning_iterations: int = 10,
    ) -> None:
        self.registry = registry or default_kernel_registry()
        self.target = cuda_target()
        self.autotune = autotune
        self.tuning_warmup = tuning_warmup
        self.tuning_iterations = tuning_iterations
        self.tuning_results: list[TuningResult] = []
        self._trace: list[PlannedRegion] = []
        self._routes: dict[
            tuple[Any, ...], tuple[PlannedRegion, KernelCandidate]
        ] = {}

    def begin_step(self) -> None:
        self._trace.clear()

    @property
    def trace(self) -> tuple[PlannedRegion, ...]:
        return tuple(self._trace)

    def run(
        self,
        kind: RegionKind,
        *args: Any,
        phase: ExecutionPhase = ExecutionPhase.GENERAL,
        attributes: Mapping[str, Any] | None = None,
        mutable_inputs: Iterable[int] | Mapping[int, str] = (),
        site: str | None = None,
        shape_bucket: Any = None,
        **kwargs: Any,
    ) -> Any:
        aliases = (
            dict(mutable_inputs)
            if isinstance(mutable_inputs, Mapping)
            else {index: f"arg{index}" for index in mutable_inputs}
        )
        attr_key = tuple(
            (str(key), repr(value))
            for key, value in sorted((attributes or {}).items())
        )
        if site is not None:
            # Model call sites already know their dynamic shape regime. This
            # key makes the hot path a dictionary lookup plus direct function
            # call; tensor contracts are materialized only on the first call
            # in a bucket.
            route_key = (
                site,
                kind,
                phase,
                self.target,
                repr(shape_bucket),
                attr_key,
            )
            contracts = None
        else:
            contracts = self._contracts(kind, phase, args, aliases)
            route_key = (
                kind,
                phase,
                self.target,
                tuple(
                    (
                        spec.tactic_shape or spec.shape,
                        spec.dtype,
                        spec.layout,
                        spec.alias_of,
                        spec.mutable,
                    )
                    for spec in contracts
                ),
                attr_key,
            )
        resolved = self._routes.get(route_key)
        if resolved is not None:
            planned, candidate = resolved
            self._trace.append(planned)
            return candidate.run(*args, **kwargs)

        if contracts is None:
            contracts = self._contracts(kind, phase, args, aliases)
        region = TileRegion.create(
            kind,
            inputs=contracts,
            phase=phase,
            target=self.target,
            attributes=attributes,
        )
        cached = self.registry.tactic_cache.get(region)
        should_tune = (
            self.autotune
            and len(self.registry.candidates(region)) > 1
            and (cached is None or "latency_ms" not in cached.parameters)
        )
        if should_tune:
            candidate = self._autotune(region, args, kwargs)
        else:
            candidate = self.registry.resolve(region)
        planned = PlannedRegion(region, candidate.name, candidate.backend)
        self._routes[route_key] = (planned, candidate)
        self._trace.append(planned)
        return candidate.run(*args, **kwargs)

    def _autotune(
        self,
        region: TileRegion,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> KernelCandidate:
        import cupy as cp

        candidates = self.registry.candidates(region)
        reference_candidate = candidates[0]
        reference = self._copy_output(reference_candidate.run(*args, **kwargs))
        cp.cuda.get_current_stream().synchronize()
        measurements = []

        for candidate in candidates:
            try:
                output = candidate.run(*args, **kwargs)
                cp.cuda.get_current_stream().synchronize()
                valid, max_error = self._outputs_close(reference, output)
                if not valid:
                    measurements.append(
                        CandidateMeasurement(
                            candidate.name,
                            candidate.backend,
                            None,
                            False,
                            max_abs_error=max_error,
                            error="output mismatch",
                        )
                    )
                    continue

                for _ in range(self.tuning_warmup):
                    candidate.run(*args, **kwargs)
                cp.cuda.get_current_stream().synchronize()
                start, end = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                for _ in range(self.tuning_iterations):
                    candidate.run(*args, **kwargs)
                end.record()
                end.synchronize()
                latency = (
                    cp.cuda.get_elapsed_time(start, end) / self.tuning_iterations
                )
                measurements.append(
                    CandidateMeasurement(
                        candidate.name,
                        candidate.backend,
                        latency,
                        True,
                        max_abs_error=max_error,
                    )
                )
            except Exception as exc:
                measurements.append(
                    CandidateMeasurement(
                        candidate.name,
                        candidate.backend,
                        None,
                        False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        valid = [item for item in measurements if item.valid]
        if not valid:
            raise RuntimeError(f"all candidates failed for {region.kind.value}")
        winner_measurement = min(valid, key=lambda item: item.latency_ms)
        winner = next(
            item for item in candidates if item.name == winner_measurement.candidate
        )
        self.registry.remember(
            region,
            winner,
            latency_ms=float(winner_measurement.latency_ms),
        )
        self.tuning_results.append(
            TuningResult(region, winner.name, tuple(measurements))
        )
        return winner

    @classmethod
    def _copy_output(cls, output: Any) -> Any:
        if isinstance(output, tuple):
            return tuple(cls._copy_output(item) for item in output)
        copy = getattr(output, "copy", None)
        return copy() if copy is not None else output

    @classmethod
    def _outputs_close(cls, reference: Any, output: Any) -> tuple[bool, float]:
        import cupy as cp

        if isinstance(reference, tuple):
            if not isinstance(output, tuple) or len(reference) != len(output):
                return False, math.inf
            results = [
                cls._outputs_close(expected, actual)
                for expected, actual in zip(reference, output, strict=True)
            ]
            return all(valid for valid, _ in results), max(
                (error for _, error in results), default=0.0
            )
        if getattr(reference, "shape", None) != getattr(output, "shape", None):
            return False, math.inf
        ref = reference.astype(cp.float32)
        actual = output.astype(cp.float32)
        error = float(cp.max(cp.abs(ref - actual)))
        scale = float(cp.max(cp.abs(ref)))
        return error <= 1e-2 + 1e-2 * scale, error

    @classmethod
    def _contracts(
        cls,
        kind: RegionKind,
        phase: ExecutionPhase,
        args: tuple[Any, ...],
        aliases: Mapping[int, str],
    ) -> list[TensorContract]:
        contracts = []
        for index, value in enumerate(args):
            if not hasattr(value, "shape") or not hasattr(value, "dtype"):
                continue
            shape = tuple(int(dim) for dim in value.shape)
            contracts.append(
                TensorContract.from_array(
                    f"arg{index}",
                    value,
                    alias_of=aliases.get(index),
                    mutable=index in aliases,
                    tactic_shape=cls._tactic_shape(kind, phase, index, shape),
                )
            )
        return contracts

    @staticmethod
    def _tactic_shape(
        kind: RegionKind,
        phase: ExecutionPhase,
        index: int,
        shape: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        """Bucket only dimensions that select tactics, never kernel arguments."""
        if kind is not RegionKind.ATTENTION or len(shape) != 4:
            return None
        # Attention tactics depend on query/KV length regimes. Context grows by
        # one per decode step; power-of-two buckets avoid planning and caching a
        # nominally different implementation for every token.
        if index not in (0, 1, 2):
            return None
        bucketed = list(shape)
        seq = bucketed[2]
        bucketed[2] = length_bucket(seq)
        return tuple(bucketed)
