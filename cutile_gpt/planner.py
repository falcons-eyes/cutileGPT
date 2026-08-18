# SPDX-License-Identifier: Apache-2.0
"""Tile-aware planning contracts shared by graph and kernel runtimes.

The graph planner owns semantic fusion boundaries, tensor lifetimes, and
backend selection.  A kernel backend still owns thread mapping, register and
shared-memory allocation, and the TMA/WGMMA schedule.  Keeping those concerns
out of :class:`TileRegion` is intentional: a region describes *what* should be
computed, not how a GPU should execute it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterable, Mapping


class ExecutionPhase(str, Enum):
    """Shape regimes whose best tactics commonly differ."""

    GENERAL = "general"
    PREFILL = "prefill"
    DECODE = "decode"


class RegionKind(str, Enum):
    """Semantic regions large enough to justify an opaque kernel boundary."""

    LINEAR = "linear"
    QK_NORM_ROPE = "qk_norm_rope"
    QK_NORM_ROPE_CACHE = "qk_norm_rope_cache"
    ATTENTION = "attention"
    LINEAR_RESIDUAL = "linear_residual"
    SWIGLU_RESIDUAL = "swiglu_residual"
    GELU_MLP_RESIDUAL = "gelu_mlp_residual"


class Backend(str, Enum):
    CUTILE = "cutile"
    CUTE = "cute"
    CUTLASS = "cutlass"
    TORCH = "torch"


def _stable_value(value: Any) -> Any:
    """Convert an attribute into deterministic JSON-compatible data."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _stable_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_stable_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class TensorContract:
    """The boundary information a graph planner may impose on a kernel."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    layout: str = "strided"
    alias_of: str | None = None
    mutable: bool = False
    tactic_shape: tuple[int, ...] | None = None

    @classmethod
    def from_array(
        cls,
        name: str,
        array: Any,
        *,
        alias_of: str | None = None,
        mutable: bool = False,
        tactic_shape: tuple[int, ...] | None = None,
    ) -> "TensorContract":
        shape = tuple(int(dim) for dim in array.shape)
        flags = getattr(array, "flags", None)
        contiguous = bool(getattr(flags, "c_contiguous", False))
        return cls(
            name=name,
            shape=shape,
            dtype=str(array.dtype),
            layout="row_major" if contiguous else "strided",
            alias_of=alias_of,
            mutable=mutable,
            tactic_shape=tactic_shape,
        )


@dataclass(frozen=True)
class TileRegion:
    """A semantic fused operation presented to a tile backend."""

    kind: RegionKind
    inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...] = ()
    phase: ExecutionPhase = ExecutionPhase.GENERAL
    target: str = "unknown"
    attributes: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def create(
        cls,
        kind: RegionKind,
        *,
        inputs: Iterable[TensorContract],
        outputs: Iterable[TensorContract] = (),
        phase: ExecutionPhase = ExecutionPhase.GENERAL,
        target: str = "unknown",
        attributes: Mapping[str, Any] | None = None,
    ) -> "TileRegion":
        attrs = tuple(
            (str(key), _stable_value(value))
            for key, value in sorted((attributes or {}).items())
        )
        return cls(kind, tuple(inputs), tuple(outputs), phase, target, attrs)

    @property
    def cache_key(self) -> str:
        def tactic_contract(spec: TensorContract) -> dict[str, Any]:
            return {
                "name": spec.name,
                "shape": spec.tactic_shape or spec.shape,
                "dtype": spec.dtype,
                "layout": spec.layout,
                "alias_of": spec.alias_of,
                "mutable": spec.mutable,
            }

        payload = {
            "kind": self.kind.value,
            "phase": self.phase.value,
            "target": self.target,
            "inputs": [tactic_contract(spec) for spec in self.inputs],
            "outputs": [tactic_contract(spec) for spec in self.outputs],
            "attributes": self.attributes,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class TacticSelection:
    """A cached backend implementation and its compile-time parameters."""

    candidate: str
    backend: Backend
    parameters: Mapping[str, int | float | str | bool] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "backend": self.backend.value,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "TacticSelection":
        return cls(
            candidate=str(value["candidate"]),
            backend=Backend(value["backend"]),
            parameters=dict(value.get("parameters", {})),
        )


class TacticCache:
    """Thread-safe tactic cache with optional explicit persistence.

    Persistence is opt-in so importing or running a model never writes to the
    user's machine unexpectedly.  Call :meth:`save` after an autotuning run.
    """

    FORMAT_VERSION = 1

    def __init__(self) -> None:
        self._entries: dict[str, TacticSelection] = {}
        self._lock = RLock()

    def get(self, region: TileRegion) -> TacticSelection | None:
        with self._lock:
            return self._entries.get(region.cache_key)

    def put(self, region: TileRegion, selection: TacticSelection) -> None:
        with self._lock:
            self._entries[region.cache_key] = selection

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with self._lock:
            payload = {
                "version": self.FORMAT_VERSION,
                "entries": {
                    key: value.to_json() for key, value in self._entries.items()
                },
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text())
        if payload.get("version") != self.FORMAT_VERSION:
            raise ValueError(
                f"unsupported tactic cache version {payload.get('version')}"
            )
        entries = {
            key: TacticSelection.from_json(value)
            for key, value in payload.get("entries", {}).items()
        }
        with self._lock:
            self._entries = entries


DEFAULT_TACTIC_CACHE = TacticCache()


def length_bucket(length: int) -> int:
    """Power-of-two bucket used for dynamic token and context dimensions."""
    if length < 1:
        raise ValueError(f"length must be positive, got {length}")
    return 1 if length == 1 else 1 << (length - 1).bit_length()


@dataclass(frozen=True)
class RegionCost:
    """Portable pre-benchmark estimate for a semantic implementation.

    It intentionally models only costs visible to the graph planner. Register
    allocation, memory swizzles, TMA, and MMA schedules remain compiler-owned
    and are resolved by measurement when estimates are close.
    """

    launches: int
    materialized_bytes: int = 0
    recompute_flops: int = 0
    workspace_bytes: int = 0

    def estimated_us(
        self,
        *,
        launch_us: float = 3.0,
        bandwidth_gbps: float = 500.0,
        compute_tflops: float = 50.0,
    ) -> float:
        memory_us = (
            (self.materialized_bytes + self.workspace_bytes)
            / (bandwidth_gbps * 1e9)
            * 1e6
        )
        compute_us = self.recompute_flops / (compute_tflops * 1e12) * 1e6
        return self.launches * launch_us + memory_us + compute_us


@dataclass(frozen=True)
class KernelCandidate:
    """One backend implementation for a semantic region."""

    name: str
    backend: Backend
    run: Callable[..., Any]
    supports: Callable[[TileRegion], bool] = lambda _region: True
    priority: int = 0
    cost: RegionCost | Callable[[TileRegion], RegionCost] | None = None

    def estimate(self, region: TileRegion) -> RegionCost | None:
        if self.cost is None:
            return None
        return self.cost(region) if callable(self.cost) else self.cost


class KernelRegistry:
    """Resolve semantic regions without leaking low-level schedules upward."""

    def __init__(self, tactic_cache: TacticCache | None = None) -> None:
        self.tactic_cache = (
            tactic_cache if tactic_cache is not None else DEFAULT_TACTIC_CACHE
        )
        self._candidates: dict[RegionKind, list[KernelCandidate]] = {}

    def register(self, kind: RegionKind, candidate: KernelCandidate) -> None:
        candidates = self._candidates.setdefault(kind, [])
        if any(existing.name == candidate.name for existing in candidates):
            raise ValueError(f"candidate {candidate.name!r} already registered")
        candidates.append(candidate)
        candidates.sort(key=lambda item: item.priority, reverse=True)

    def candidates(self, region: TileRegion) -> tuple[KernelCandidate, ...]:
        return tuple(
            candidate
            for candidate in self._candidates.get(region.kind, ())
            if candidate.supports(region)
        )

    def resolve(self, region: TileRegion) -> KernelCandidate:
        candidates = self.candidates(region)
        if not candidates:
            raise LookupError(
                f"no kernel candidate supports {region.kind.value} on {region.target}"
            )

        cached = self.tactic_cache.get(region)
        if cached is not None:
            for candidate in candidates:
                if (
                    candidate.name == cached.candidate
                    and candidate.backend == cached.backend
                ):
                    return candidate

        estimated = [
            (candidate.estimate(region), candidate) for candidate in candidates
        ]
        costed = [(cost, candidate) for cost, candidate in estimated if cost]
        if costed:
            chosen = min(
                costed,
                key=lambda item: (item[0].estimated_us(), -item[1].priority),
            )[1]
        else:
            chosen = candidates[0]
        self.tactic_cache.put(
            region,
            TacticSelection(chosen.name, chosen.backend),
        )
        return chosen

    def remember(
        self,
        region: TileRegion,
        candidate: KernelCandidate,
        **parameters: int | float | str | bool,
    ) -> None:
        """Persist an explicitly measured selection for future processes."""
        self.tactic_cache.put(
            region,
            TacticSelection(candidate.name, candidate.backend, parameters),
        )

    def execute(self, region: TileRegion, *args: Any, **kwargs: Any) -> Any:
        return self.resolve(region).run(*args, **kwargs)


def cuda_target() -> str:
    """Return a stable target string without making CuPy a planner dependency."""
    try:
        import cupy as cp

        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        major = int(props.get("major", 0))
        minor = int(props.get("minor", 0))
        name = props.get("name", "cuda")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        return f"sm_{major}{minor}:{name}"
    except Exception:
        return "unknown"
