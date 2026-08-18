#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare staged GELU-MLP fusion with three separate kernel calls."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--intermediate", type=int, default=1536)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    import cupy as cp

    from cutile_gpt import (
        cutile_fused_mlp,
        cutile_gelu,
        cutile_linear_bias,
    )

    cp.random.seed(0)
    x = cp.random.standard_normal((args.rows, args.hidden), dtype=cp.float32)
    w_fc = cp.random.standard_normal(
        (args.intermediate, args.hidden), dtype=cp.float32
    ) * 0.02
    b_fc = cp.zeros(args.intermediate, dtype=cp.float32)
    w_proj = cp.random.standard_normal(
        (args.hidden, args.intermediate), dtype=cp.float32
    ) * 0.02
    b_proj = cp.zeros(args.hidden, dtype=cp.float32)

    def staged():
        return cutile_fused_mlp(x, w_fc, b_fc, w_proj, b_proj)

    def separate():
        hidden = cutile_linear_bias(x, w_fc, b_fc)
        hidden = cutile_gelu(hidden)
        return cutile_linear_bias(hidden, w_proj, b_proj)

    def measure(fn) -> float:
        for _ in range(args.warmup):
            fn()
        cp.cuda.get_current_stream().synchronize()
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        for _ in range(args.iterations):
            fn()
        end.record()
        end.synchronize()
        return cp.cuda.get_elapsed_time(start, end) / args.iterations

    staged_ms = measure(staged)
    separate_ms = measure(separate)
    error = float(cp.max(cp.abs(staged() - separate())))
    cp.cuda.get_current_stream().synchronize()

    print(f"shape: ({args.rows}, {args.hidden}) -> {args.intermediate}")
    print(f"staged:  {staged_ms:.5f} ms")
    print(f"separate: {separate_ms:.5f} ms")
    print(f"speedup:  {separate_ms / staged_ms:.3f}x")
    print(f"max error: {error:g}")


if __name__ == "__main__":
    main()
