# SPDX-License-Identifier: Apache-2.0
"""
Linear kernel parameter sweep for finding optimal num_ctas/occupancy.

Tests different combinations to find the best performance for the target GPU.
"""

import math
import torch
import cuda.tile as ct
import time

ConstInt = ct.Constant[int]

GROUP_SIZE_M = 8


def swizzle_2d(M, N, tm, tn):
    """Get swizzled 2D block indices for better L2 locality."""
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


# Generate kernels with different configurations
def make_matmul_kernel(num_ctas_val, occupancy_val):
    """Factory function to create kernels with specific parameters."""

    @ct.kernel(num_ctas=num_ctas_val, occupancy=occupancy_val)
    def matmul_bias_kernel(A, B, bias, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
        M = A.shape[0]
        N = B.shape[1]
        bid_m, bid_n = swizzle_2d(M, N, tm, tn)
        num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
        acc = ct.full((tm, tn), 0, dtype=ct.float32)
        zero_pad = ct.PaddingMode.ZERO
        dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

        for k in range(num_tiles_k):
            a = ct.load(A, index=(bid_m, k), shape=(tm, tk),
                        padding_mode=zero_pad, latency=4).astype(dtype)
            b = ct.load(B, index=(k, bid_n), shape=(tk, tn),
                        padding_mode=zero_pad, latency=4).astype(dtype)
            acc = ct.mma(a, b, acc)

        b_tile = ct.load(bias, index=(bid_n,), shape=(tn,),
                         padding_mode=zero_pad, latency=2)
        acc = acc + b_tile
        ct.store(C, index=(bid_m, bid_n), tile=acc.astype(C.dtype))

    return matmul_bias_kernel


def benchmark_kernel(kernel, x_2d, weight_t, bias, output, tm, tn, tk, grid, warmup=5, iters=20):
    """Benchmark a kernel configuration."""
    stream = torch.cuda.current_stream()

    # Warmup
    for _ in range(warmup):
        ct.launch(stream, grid, kernel, (x_2d, weight_t, bias, output, tm, tn, tk))
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        ct.launch(stream, grid, kernel, (x_2d, weight_t, bias, output, tm, tn, tk))
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def run_sweep():
    """Run parameter sweep for linear kernel."""
    print("=" * 60)
    print("Linear Kernel Parameter Sweep")
    print("=" * 60)

    # Test sizes matching our model configs
    configs = [
        ("small", 2, 64, 64, 256),   # batch*seq=128, in=64, out=256
        ("medium", 2, 128, 128, 512), # batch*seq=256, in=128, out=512
        ("large", 2, 256, 256, 1024), # batch*seq=512, in=256, out=1024
    ]

    # Parameters to sweep
    num_ctas_options = [1, 2, 4]
    occupancy_options = [1, 2, 4]

    for name, batch, seq, in_feat, out_feat in configs:
        print(f"\n--- Config: {name} (M={batch*seq}, K={in_feat}, N={out_feat}) ---")

        # Setup tensors
        x = torch.randn(batch, seq, in_feat, dtype=torch.float32, device='cuda')
        weight = torch.randn(out_feat, in_feat, dtype=torch.float32, device='cuda')
        bias_tensor = torch.randn(out_feat, dtype=torch.float32, device='cuda')

        x_2d = x.reshape(-1, in_feat).contiguous()
        weight_t = weight.t().contiguous()
        M, N = x_2d.shape[0], out_feat
        output = torch.empty(M, N, dtype=x.dtype, device=x.device)

        tm, tn, tk = 32, 32, 32
        grid_m = math.ceil(M / tm)
        grid_n = math.ceil(N / tn)
        grid = (grid_m * grid_n, 1, 1)

        # PyTorch baseline
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(5):  # warmup
            _ = torch.nn.functional.linear(x, weight, bias_tensor)
        torch.cuda.synchronize()

        start.record()
        for _ in range(20):
            _ = torch.nn.functional.linear(x, weight, bias_tensor)
        end.record()
        torch.cuda.synchronize()
        pytorch_time = start.elapsed_time(end) / 20

        print(f"PyTorch baseline: {pytorch_time:.4f} ms")
        print()

        best_time = float('inf')
        best_config = None

        for num_ctas in num_ctas_options:
            for occupancy in occupancy_options:
                try:
                    kernel = make_matmul_kernel(num_ctas, occupancy)
                    time_ms = benchmark_kernel(kernel, x_2d, weight_t, bias_tensor,
                                               output, tm, tn, tk, grid)
                    speedup = pytorch_time / time_ms

                    print(f"  num_ctas={num_ctas}, occupancy={occupancy}: "
                          f"{time_ms:.4f} ms ({speedup:.2f}x vs PyTorch)")

                    if time_ms < best_time:
                        best_time = time_ms
                        best_config = (num_ctas, occupancy)
                except Exception as e:
                    print(f"  num_ctas={num_ctas}, occupancy={occupancy}: FAILED - {e}")

        if best_config:
            print(f"\n  Best: num_ctas={best_config[0]}, occupancy={best_config[1]} "
                  f"-> {best_time:.4f} ms ({pytorch_time/best_time:.2f}x vs PyTorch)")


if __name__ == "__main__":
    run_sweep()
