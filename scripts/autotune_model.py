#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tune representative linear tile shapes for one real checkpoint.

The graph layer supplies only semantic shapes.  ``cuda.tile`` still compiles
each candidate and owns its thread, register, shared-memory, TMA, and WGMMA
mapping.  The winning compile-time tile parameters are saved by GPU, dtype,
shape, and prefill/decode phase.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_transformer import parse_lengths, resolve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--rows", type=parse_lengths, default=[1, 128, 512])
    parser.add_argument(
        "--cache", type=Path, default=Path(".cutile-gpt-tactics.json")
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        help="limit unique weight shapes for a quick smoke run",
    )
    args = parser.parse_args()

    import cupy as cp

    from cutile_gpt import autotune_linear
    from cutile_gpt.models.transformer import TransformerLM
    from cutile_gpt.planner import DEFAULT_TACTIC_CACHE

    if args.cache.exists():
        DEFAULT_TACTIC_CACHE.load(args.cache)

    path = resolve(args.model, not args.no_download)
    model = TransformerLM.from_pretrained(path)

    unique: dict[tuple[str, int, int], cp.ndarray] = {}
    for weight in model.weights.values():
        if weight.ndim != 2:
            continue
        n, k = map(int, weight.shape)
        unique.setdefault((str(weight.dtype), n, k), weight)

    shapes = list(unique.items())
    if args.max_shapes is not None:
        shapes = shapes[: args.max_shapes]

    total = len(shapes) * len(args.rows)
    completed = 0
    print(f"Tuning {len(shapes)} unique linear shapes at rows={args.rows}")
    for (dtype, n, k), weight in shapes:
        for rows in args.rows:
            completed += 1
            x = cp.empty((rows, k), dtype=weight.dtype)
            result = autotune_linear(x, weight, quiet=True)
            best = result.best.config
            print(
                f"[{completed:>2}/{total}] {dtype} ({rows}, {k}) x "
                f"({n}, {k}).T -> tile=({best.tm}, {best.tn}, {best.tk})"
            )

    DEFAULT_TACTIC_CACHE.save(args.cache)
    print(f"Saved {len(DEFAULT_TACTIC_CACHE)} tactics to {args.cache}")


if __name__ == "__main__":
    main()
