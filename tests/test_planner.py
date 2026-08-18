# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the graph/tile planning boundary."""

from cutile_gpt.planner import (
    Backend,
    ExecutionPhase,
    KernelCandidate,
    KernelRegistry,
    RegionCost,
    RegionKind,
    TacticCache,
    TacticSelection,
    TensorContract,
    TileRegion,
)
from cutile_gpt.regions import TileRuntime


def region(
    *,
    rows: int = 1,
    phase: ExecutionPhase = ExecutionPhase.DECODE,
    attributes=None,
) -> TileRegion:
    return TileRegion.create(
        RegionKind.LINEAR,
        inputs=(
            TensorContract("x", (rows, 128), "float16", "row_major"),
            TensorContract("weight", (256, 128), "float16", "row_major"),
        ),
        phase=phase,
        target="sm_100:test",
        attributes=attributes or {"weight_layout": "out_in"},
    )


def test_region_cache_key_is_stable_and_shape_specific():
    first = region(attributes={"z": 3, "a": True})
    reordered = region(attributes={"a": True, "z": 3})

    assert first.cache_key == reordered.cache_key
    assert first.cache_key != region(rows=128).cache_key
    assert first.cache_key != region(phase=ExecutionPhase.PREFILL).cache_key


def test_registry_respects_cached_tactic_after_new_candidate_is_added():
    cache = TacticCache()
    registry = KernelRegistry(cache)
    registry.register(
        RegionKind.LINEAR,
        KernelCandidate("cutile.baseline", Backend.CUTILE, lambda x: x, priority=1),
    )
    selected = registry.resolve(region())
    assert selected.name == "cutile.baseline"

    registry.register(
        RegionKind.LINEAR,
        KernelCandidate("cutlass.new", Backend.CUTLASS, lambda x: x, priority=9),
    )
    assert registry.resolve(region()).name == "cutile.baseline"


def test_registry_filters_unsupported_candidates():
    registry = KernelRegistry(TacticCache())
    registry.register(
        RegionKind.LINEAR,
        KernelCandidate(
            "cutlass.prefill",
            Backend.CUTLASS,
            lambda x: x,
            supports=lambda item: item.phase is ExecutionPhase.PREFILL,
            priority=10,
        ),
    )
    registry.register(
        RegionKind.LINEAR,
        KernelCandidate("cutile.any", Backend.CUTILE, lambda x: x),
    )

    assert registry.resolve(region()).name == "cutile.any"
    assert (
        registry.resolve(region(phase=ExecutionPhase.PREFILL)).name
        == "cutlass.prefill"
    )


def test_registry_cost_model_accounts_for_launches_before_priority():
    registry = KernelRegistry(TacticCache())
    registry.register(
        RegionKind.LINEAR,
        KernelCandidate(
            "many.launches",
            Backend.CUTILE,
            lambda x: x,
            priority=100,
            cost=RegionCost(launches=5),
        ),
    )
    registry.register(
        RegionKind.LINEAR,
        KernelCandidate(
            "one.launch",
            Backend.CUTILE,
            lambda x: x,
            cost=RegionCost(launches=1, materialized_bytes=1024),
        ),
    )

    assert registry.resolve(region()).name == "one.launch"


def test_tactic_cache_round_trip(tmp_path):
    item = region()
    cache = TacticCache()
    cache.put(
        item,
        TacticSelection(
            "cutile.matmul_weight",
            Backend.CUTILE,
            {"tm": 16, "tn": 128, "tk": 64},
        ),
    )
    path = tmp_path / "tactics.json"
    cache.save(path)

    restored = TacticCache()
    restored.load(path)

    assert restored.get(item) == cache.get(item)


def test_tensor_contract_records_alias_and_mutation():
    output = TensorContract(
        "kv_cache", (1, 8, 4096, 128), "bfloat16", "strided",
        alias_of="kv_cache", mutable=True,
    )

    assert output.alias_of == "kv_cache"
    assert output.mutable


class _Flags:
    c_contiguous = True


class _Array:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = "float16"
        self.flags = _Flags()


def test_decode_attention_uses_context_buckets_for_tactic_selection():
    cache = TacticCache()
    registry = KernelRegistry(cache)
    registry.register(
        RegionKind.ATTENTION,
        KernelCandidate(
            "cutile.attention", Backend.CUTILE, lambda q, k, v, heads: q
        ),
    )
    runtime = TileRuntime(registry)
    q = _Array((1, 8, 1, 64))

    runtime.run(
        RegionKind.ATTENTION,
        q,
        _Array((1, 2, 129, 64)),
        _Array((1, 2, 129, 64)),
        8,
        phase=ExecutionPhase.DECODE,
    )
    runtime.run(
        RegionKind.ATTENTION,
        q,
        _Array((1, 2, 200, 64)),
        _Array((1, 2, 200, 64)),
        8,
        phase=ExecutionPhase.DECODE,
    )

    assert len(cache) == 1
