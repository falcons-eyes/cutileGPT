"""
Test different tile sizes to find optimal configuration.

Current: 64x64 tiles
Testing: 32x32, 64x64, 128x128
"""

import cupy as cp
import cuda.tile as ct
import math
import time
from cutile_gpt.kernels.attention import cutile_causal_attention, cupy_causal_attention


def benchmark_tile_size(q, k, v, n_head, tile_m, tile_n, iterations=100):
    """Benchmark specific tile size."""
    from cutile_gpt.kernels.attention import causal_attention_kernel

    batch, n_head_val, seq_len, head_dim = q.shape
    qk_scale = 1.0 / math.sqrt(head_dim)
    out = cp.empty_like(q)

    grid_x = math.ceil(seq_len / tile_m)
    grid_y = batch * n_head_val

    # Warmup
    for _ in range(20):
        ct.launch(
            cp.cuda.get_current_stream(),
            (grid_x, grid_y, 1),
            causal_attention_kernel,
            (q, k, v, out, qk_scale, head_dim, n_head_val, tile_m, tile_n)
        )
    cp.cuda.Stream.null.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        ct.launch(
            cp.cuda.get_current_stream(),
            (grid_x, grid_y, 1),
            causal_attention_kernel,
            (q, k, v, out, qk_scale, head_dim, n_head_val, tile_m, tile_n)
        )
    cp.cuda.Stream.null.synchronize()

    avg_time = (time.time() - start) / iterations * 1000
    return avg_time, out


if __name__ == "__main__":
    print("=== Testing Different Tile Sizes ===\n")

    # Test configurations
    configs = [
        (2, 4, 128, 64),   # Small: batch=2, heads=4, seq=128
        (4, 4, 256, 64),   # Medium: seq=256
        (2, 4, 512, 64),   # Large: seq=512
    ]

    tile_sizes = [
        (32, 32),
        (64, 64),
        (128, 128),
    ]

    for batch, n_head, seq_len, head_dim in configs:
        print(f"\nConfig: batch={batch}, heads={n_head}, seq={seq_len}, head_dim={head_dim}")
        print("-" * 70)

        # Generate data
        q = cp.random.randn(batch, n_head, seq_len, head_dim, dtype=cp.float32)
        k = cp.random.randn(batch, n_head, seq_len, head_dim, dtype=cp.float32)
        v = cp.random.randn(batch, n_head, seq_len, head_dim, dtype=cp.float32)

        # Reference
        y_ref = cupy_causal_attention(q, k, v, n_head)

        results = []
        for tile_m, tile_n in tile_sizes:
            try:
                avg_time, y_out = benchmark_tile_size(q, k, v, n_head, tile_m, tile_n)

                # Check correctness
                max_diff = cp.abs(y_out - y_ref).max()

                results.append({
                    'tile': f"{tile_m}x{tile_n}",
                    'time': avg_time,
                    'diff': max_diff
                })

                print(f"  {tile_m:3d}x{tile_n:<3d}: {avg_time:.4f} ms  (err: {max_diff:.6f})")
            except Exception as e:
                print(f"  {tile_m:3d}x{tile_n:<3d}: FAILED - {e}")

        # Find best
        if results:
            best = min(results, key=lambda x: x['time'])
            print(f"\n  ✅ Best: {best['tile']} at {best['time']:.4f} ms")

    print("\n=== Tile Size Test Complete ===")
