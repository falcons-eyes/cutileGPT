#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Profile cutileGPT performance using NVIDIA profiling tools.

This script is designed to work with:
- nsys (Nsight Systems): System-wide performance analysis
- ncu (Nsight Compute): Detailed kernel-level metrics

Usage:
    # System-wide profile with nsys
    nsys profile -o cutile_profile --stats=true python profile_performance.py --tool nsys

    # Kernel-level profile with ncu
    ncu --set full -o cutile_kernel_profile python profile_performance.py --tool ncu

    # Or just run benchmarks
    python profile_performance.py
"""

import argparse
import time
import cupy as cp
import torch
from cutile_gpt.model import CutileGPT, CutileGPTConfig


def benchmark_forward_pass(model, idx, warmup=10, iterations=100):
    """
    Benchmark forward pass with accurate GPU timing.

    Args:
        model: CutileGPT model
        idx: Input tokens (CuPy array)
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations

    Returns:
        Dictionary with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        _, _ = model(idx)
    cp.cuda.Stream.null.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        _, _ = model(idx)
        end.record()
        end.synchronize()

        elapsed = cp.cuda.get_elapsed_time(start, end)  # milliseconds
        times.append(elapsed)

    times = cp.array(times)
    return {
        'mean': float(cp.mean(times)),
        'std': float(cp.std(times)),
        'min': float(cp.min(times)),
        'max': float(cp.max(times)),
        'median': float(cp.median(times)),
    }


def profile_model_configs():
    """Profile different model configurations."""
    configs = {
        'gpt_nano': CutileGPTConfig.gpt_nano(),
        'gpt_micro': CutileGPTConfig.gpt_micro(),
        'gpt_mini': CutileGPTConfig.gpt_mini(),
        'gpt_tile_small': CutileGPTConfig.gpt_tile_small(),
        'gpt_tile_medium': CutileGPTConfig.gpt_tile_medium(),
        'gpt_tile_large': CutileGPTConfig.gpt_tile_large(),
    }

    results = {}

    for name, config in configs.items():
        print(f"\n{'='*80}")
        print(f"Profiling: {name}")
        print(f"  Layers: {config.n_layer}, Heads: {config.n_head}, Embedding: {config.n_embd}")
        print(f"{'='*80}")

        # Create model
        model = CutileGPT(config, use_fused_mlp=False)

        # Create input
        batch_size = 4
        seq_len = 64
        idx = cp.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=cp.int64)

        # Benchmark
        stats = benchmark_forward_pass(model, idx, warmup=10, iterations=100)

        results[name] = stats

        print(f"\nForward Pass Timing:")
        print(f"  Mean:   {stats['mean']:.4f} ms")
        print(f"  Std:    {stats['std']:.4f} ms")
        print(f"  Min:    {stats['min']:.4f} ms")
        print(f"  Max:    {stats['max']:.4f} ms")
        print(f"  Median: {stats['median']:.4f} ms")

        # Estimate throughput
        tokens_per_second = (batch_size * seq_len * 1000) / stats['mean']
        print(f"\nThroughput: {tokens_per_second:.0f} tokens/sec")

    return results


def profile_kernel_analysis():
    """
    Profile for kernel-level analysis (use with ncu).

    This runs a smaller workload optimized for detailed kernel profiling.
    """
    print("\n" + "="*80)
    print("Kernel-Level Profile (for use with ncu)")
    print("="*80)

    # Use tile-optimized config for best kernel performance
    config = CutileGPTConfig.gpt_tile_medium()
    model = CutileGPT(config, use_fused_mlp=False)

    # Small batch for detailed profiling
    batch_size = 1
    seq_len = 32
    idx = cp.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=cp.int64)

    print(f"\nRunning forward pass...")
    print(f"  Batch: {batch_size}, Seq Len: {seq_len}")
    print(f"  Model: {config.n_layer} layers, {config.n_embd} dims")

    # Single forward pass for ncu to analyze
    start_time = time.time()
    logits, _ = model(idx)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start_time

    print(f"\nForward pass completed in {elapsed*1000:.2f} ms")
    print(f"Output shape: {logits.shape}")
    print("\nUse ncu to see detailed kernel metrics:")
    print("  ncu --set full -o kernel_profile python profile_performance.py --tool ncu")


def profile_system_wide():
    """
    Profile for system-wide analysis (use with nsys).

    This runs a realistic workload for system-level profiling.
    """
    print("\n" + "="*80)
    print("System-Wide Profile (for use with nsys)")
    print("="*80)

    # Use realistic config
    config = CutileGPTConfig.gpt_tile_large()
    model = CutileGPT(config, use_fused_mlp=False)

    # Realistic batch
    batch_size = 8
    seq_len = 128
    idx = cp.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=cp.int64)

    print(f"\nRunning benchmark...")
    print(f"  Batch: {batch_size}, Seq Len: {seq_len}")
    print(f"  Model: {config.n_layer} layers, {config.n_embd} dims")

    # Run multiple iterations for nsys timeline
    iterations = 20
    print(f"\nExecuting {iterations} forward passes...")

    for i in range(iterations):
        logits, _ = model(idx)
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{iterations}")

    cp.cuda.Stream.null.synchronize()
    print("\nBenchmark completed!")
    print("\nUse nsys to see timeline and GPU utilization:")
    print("  nsys profile -o profile --stats=true python profile_performance.py --tool nsys")


def main():
    """Main profiling function."""
    parser = argparse.ArgumentParser(description='Profile cutileGPT performance')
    parser.add_argument('--tool', choices=['nsys', 'ncu', 'benchmark'], default='benchmark',
                      help='Profiling tool mode')
    args = parser.parse_args()

    print("="*80)
    print("cutileGPT Performance Profiling")
    print("="*80)
    print(f"Mode: {args.tool}")
    print(f"CuPy version: {cp.__version__}")
    print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")

    # Get GPU info
    device = cp.cuda.Device()
    device_props = cp.cuda.runtime.getDeviceProperties(device.id)
    print(f"GPU: {device_props['name'].decode()}")
    print(f"Compute Capability: {device.compute_capability}")
    print("="*80)

    if args.tool == 'nsys':
        profile_system_wide()
    elif args.tool == 'ncu':
        profile_kernel_analysis()
    else:
        # Run comprehensive benchmark
        results = profile_model_configs()

        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"{'Config':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'Throughput (tok/s)':<20}")
        print("-"*80)

        for name, stats in results.items():
            throughput = (4 * 64 * 1000) / stats['mean']  # batch=4, seq=64
            print(f"{name:<20} {stats['mean']:<12.4f} {stats['std']:<12.4f} {throughput:<20.0f}")

        print("\n" + "="*80)
        print("To run detailed profiling:")
        print("  nsys: nsys profile -o profile --stats=true python profile_performance.py --tool nsys")
        print("  ncu:  ncu --set full -o kernel_profile python profile_performance.py --tool ncu")
        print("="*80)


if __name__ == "__main__":
    main()
