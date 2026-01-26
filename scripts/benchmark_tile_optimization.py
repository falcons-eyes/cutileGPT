"""
Benchmark the tile size optimization impact on full model.

Before: 64x64 tiles
After: 32x32 tiles

Expected: ~30-40% speedup on attention layer
"""

import cupy as cp
import time
from cutile_gpt.model import CutileGPT, CutileGPTConfig


def benchmark_model(config, batch_size=4, seq_len=128, iterations=100):
    """Benchmark model with given config."""
    model = CutileGPT(config, use_fused_mlp=False)

    # Create input
    idx = cp.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=cp.int64)

    # Warmup
    for _ in range(10):
        _ = model.forward(idx)
    cp.cuda.Stream.null.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = model.forward(idx)
    cp.cuda.Stream.null.synchronize()

    avg_time = (time.time() - start) / iterations * 1000
    return avg_time


if __name__ == "__main__":
    print("=== Benchmark: Tile Size Optimization (32x32 vs 64x64) ===\n")

    configs = {
        'gpt_tile_small': CutileGPTConfig.gpt_tile_small(),
        'gpt_tile_medium': CutileGPTConfig.gpt_tile_medium(),
        'gpt_tile_large': CutileGPTConfig.gpt_tile_large(),
    }

    batch_size = 4
    seq_len = 128

    print(f"Config: batch={batch_size}, seq={seq_len}\n")
    print(f"{'Model':<20} | {'Time (ms)':<10} | {'Throughput (tok/s)':<15}")
    print("-" * 55)

    for name, config in configs.items():
        avg_time = benchmark_model(config, batch_size, seq_len, iterations=100)
        throughput = (batch_size * seq_len) / (avg_time / 1000)

        print(f"{name:<20} | {avg_time:>8.3f} ms | {throughput:>13,.0f}")

    print("\n=== Benchmark Complete ===")
    print("\nNote: This uses 32x32 tiles (optimized)")
    print("Compare with previous profiling_data.json (64x64 tiles)")
