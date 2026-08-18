#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure fused and compositional candidates on real model regions."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_transformer import parse_lengths, resolve


def print_new_results(runtime, start: int) -> int:
    for result in runtime.tuning_results[start:]:
        print(f"\n{result.region.phase.value} · {result.region.kind.value}")
        for item in result.measurements:
            if item.valid:
                marker = "*" if item.candidate == result.winner else " "
                print(
                    f" {marker} {item.candidate:<42} "
                    f"{item.latency_ms:.5f} ms  error={item.max_abs_error:g}"
                )
            else:
                print(f" ! {item.candidate:<42} rejected: {item.error}")
    return len(runtime.tuning_results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--prefill", type=parse_lengths, default=[128])
    parser.add_argument("--decode", type=parse_lengths, default=[128])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--cache", type=Path, default=Path(".cutile-gpt-tactics.json")
    )
    args = parser.parse_args()

    import cupy as cp

    from cutile_gpt.models.transformer import TransformerLM
    from cutile_gpt.planner import DEFAULT_TACTIC_CACHE
    from cutile_gpt.regions import TileRuntime

    if args.cache.exists():
        DEFAULT_TACTIC_CACHE.load(args.cache)

    path = resolve(args.model, not args.no_download)
    model = TransformerLM.from_pretrained(path)
    model.tile_runtime = TileRuntime(
        autotune=True,
        tuning_warmup=args.warmup,
        tuning_iterations=args.iterations,
    )
    vocab = model.arch.vocab_size
    printed = 0

    for sequence in args.prefill:
        tokens = cp.random.randint(
            0, vocab, (args.batch, sequence), dtype=cp.int32
        )
        model.forward(tokens, last_token_only=True)
        cp.cuda.get_current_stream().synchronize()
        printed = print_new_results(model.tile_runtime, printed)

    for context in args.decode:
        cache = model.new_cache(
            batch=args.batch,
            max_seq_len=context + 2,
        )
        prefix = cp.random.randint(
            0, vocab, (args.batch, context), dtype=cp.int32
        )
        model.forward(prefix, cache=cache, last_token_only=True)
        printed = print_new_results(model.tile_runtime, printed)

        token = cp.random.randint(0, vocab, (args.batch, 1), dtype=cp.int32)
        model.forward(token, cache=cache, last_token_only=True)
        cp.cuda.get_current_stream().synchronize()
        printed = print_new_results(model.tile_runtime, printed)

    DEFAULT_TACTIC_CACHE.save(args.cache)
    print(f"\nSaved {len(DEFAULT_TACTIC_CACHE)} tactics to {args.cache}")


if __name__ == "__main__":
    main()
