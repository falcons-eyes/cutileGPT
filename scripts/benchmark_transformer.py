#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark real-checkpoint prefill and cached decode separately.

Examples:
    uv run python scripts/benchmark_transformer.py Qwen/Qwen3-0.6B
    uv run python scripts/benchmark_transformer.py /models/qwen --no-download --pytorch
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def resolve(target: str, allow_download: bool) -> Path:
    path = Path(target)
    if path.is_dir():
        return path
    if not allow_download:
        raise SystemExit(f"{target} is not a directory and --no-download is set")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            target,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
    )


def parse_lengths(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def cupy_ms(cp, fn, iterations: int) -> float:
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / iterations


def torch_ms(torch, fn, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def print_plan(model) -> None:
    """Print semantic regions, not compiler-owned thread schedules."""
    counts = Counter(
        (
            item.region.phase.value,
            item.region.kind.value,
            item.backend.value,
            item.candidate,
        )
        for item in model.tile_runtime.trace
    )
    print("\nSelected tile regions")
    print("| phase | region | backend | tactic | count |")
    print("|---|---|---|---|---:|")
    for (phase, region, backend, tactic), count in counts.items():
        print(f"| {phase} | {region} | {backend} | {tactic} | {count} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prefill", type=parse_lengths, default=[128, 512, 2048])
    parser.add_argument("--decode", type=parse_lengths, default=[128, 512, 2048])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--pytorch", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--show-plan", action="store_true")
    parser.add_argument(
        "--tactic-cache",
        type=Path,
        help="load and update an explicit JSON tactic cache",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    import cupy as cp

    from cutile_gpt import capture_forward
    from cutile_gpt.models.transformer import TransformerLM
    from cutile_gpt.planner import DEFAULT_TACTIC_CACHE

    if args.tactic_cache and args.tactic_cache.exists():
        DEFAULT_TACTIC_CACHE.load(args.tactic_cache)

    path = resolve(args.model, not args.no_download)
    model = TransformerLM.from_pretrained(path)
    vocab = model.arch.vocab_size
    results: list[dict] = []

    print(f"{args.model}\n  {model}")
    print("\nPrefill (last-token logits)")
    print("| sequence | eager ms | graph ms | graph/eager | tokens/s |")
    print("|---:|---:|---:|---:|---:|")
    for seq in args.prefill:
        idx = cp.random.randint(0, vocab, (args.batch, seq), dtype=cp.int32)
        def eager():
            return model.forward(idx, last_token_only=True)
        for _ in range(args.warmup):
            eager()
        cp.cuda.get_current_stream().synchronize()
        eager_latency = cupy_ms(cp, eager, args.iterations)

        captured = capture_forward(
            model,
            idx,
            warmup=args.warmup,
            forward_kwargs={"last_token_only": True},
        )
        graph_latency = cupy_ms(cp, lambda: captured.replay(idx), args.iterations)
        throughput = args.batch * seq * 1000 / eager_latency
        print(
            f"| {seq} | {eager_latency:.3f} | {graph_latency:.3f} | "
            f"{eager_latency / graph_latency:.3f}x | {throughput:,.0f} |"
        )
        results.extend(
            {
                "backend": backend,
                "phase": "prefill",
                "batch": args.batch,
                "sequence": seq,
                "latency_ms": latency,
            }
            for backend, latency in (
                ("cutile-eager", eager_latency),
                ("cutile-cudagraph", graph_latency),
            )
        )

    print("\nCached decode (average across advancing context)")
    print("| starting context | ms/token | tokens/s |")
    print("|---:|---:|---:|")
    for context in args.decode:
        cache = model.new_cache(
            batch=args.batch,
            max_seq_len=context + args.warmup + args.iterations + 1,
        )
        prefix = cp.random.randint(
            0, vocab, (args.batch, context), dtype=cp.int32
        )
        model.forward(prefix, cache=cache, last_token_only=True)
        token = cp.random.randint(0, vocab, (args.batch, 1), dtype=cp.int32)
        for _ in range(args.warmup):
            model.forward(token, cache=cache, last_token_only=True)
        cp.cuda.get_current_stream().synchronize()
        latency = cupy_ms(
            cp,
            lambda: model.forward(token, cache=cache, last_token_only=True),
            args.iterations,
        )
        print(f"| {context} | {latency:.3f} | {args.batch * 1000 / latency:,.0f} |")
        results.append(
            {
                "backend": "cutile-eager",
                "phase": "decode",
                "batch": args.batch,
                "sequence": context,
                "latency_ms": latency,
            }
        )

    if args.show_plan:
        print_plan(model)

    if args.pytorch:
        import torch
        from transformers import AutoModelForCausalLM, StaticCache

        hf = AutoModelForCausalLM.from_pretrained(path, dtype="auto").cuda().eval()
        if args.torch_compile:
            # Dynamic KV-cache mutation conflicts with reduce-overhead's CUDA
            # Graph output-lifetime rules. Keep Inductor fusion/autotuning but
            # disable its internal graph capture for a valid advancing decode.
            hf = torch.compile(hf, mode="max-autotune-no-cudagraphs")
        label = "torch-compile-no-graph" if args.torch_compile else "torch-eager"
        print(f"\nPyTorch comparison ({label})")
        print("| phase | sequence | ms |")
        print("|---|---:|---:|")
        with torch.no_grad():
            for seq in args.prefill:
                idx = torch.randint(0, vocab, (args.batch, seq), device="cuda")
                def fn():
                    return hf(idx, use_cache=False).logits[:, -1]
                for _ in range(args.warmup):
                    fn()
                torch.cuda.synchronize()
                latency = torch_ms(torch, fn, args.iterations)
                print(f"| prefill | {seq} | {latency:.3f} |")
                results.append(
                    {"backend": label, "phase": "prefill", "batch": args.batch,
                     "sequence": seq, "latency_ms": latency}
                )

            for context in args.decode:
                prefix = torch.randint(
                    0, vocab, (args.batch, context), device="cuda"
                )
                if args.torch_compile:
                    state = StaticCache(
                        config=hf.config,
                        max_cache_len=context + args.warmup + args.iterations + 1,
                    )
                    hf(prefix, past_key_values=state, use_cache=True)
                else:
                    state = hf(prefix, use_cache=True).past_key_values
                token = torch.randint(0, vocab, (args.batch, 1), device="cuda")

                def decode_step():
                    nonlocal state
                    if args.torch_compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    output = hf(token, past_key_values=state, use_cache=True)
                    state = output.past_key_values
                    return output.logits

                for _ in range(args.warmup):
                    decode_step()
                torch.cuda.synchronize()
                latency = torch_ms(torch, decode_step, args.iterations)
                print(f"| decode | {context} | {latency:.3f} |")
                results.append(
                    {"backend": label, "phase": "decode", "batch": args.batch,
                     "sequence": context, "latency_ms": latency}
                )

    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nWrote {args.json}")
    if args.tactic_cache:
        DEFAULT_TACTIC_CACHE.save(args.tactic_cache)
        print(
            f"Wrote {len(DEFAULT_TACTIC_CACHE)} tactics to {args.tactic_cache}"
        )


if __name__ == "__main__":
    main()
