# SPDX-License-Identifier: Apache-2.0
"""
Benchmarking Utilities

Tools for measuring performance of cutile GPT models.
"""

import time
from typing import Dict, Callable, Any
import cupy as cp


def benchmark_cupy(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    warmup: int = 5,
    iterations: int = 20
) -> Dict[str, float]:
    """
    Benchmark a CuPy-based function.

    Args:
        func: Function to benchmark
        args: Positional arguments
        kwargs: Keyword arguments
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        Dictionary with timing statistics
    """
    kwargs = kwargs or {}

    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    cp.cuda.Device().synchronize()

    # Timed iterations
    times = []
    for _ in range(iterations):
        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        _ = func(*args, **kwargs)
        end.record()
        end.synchronize()

        times.append(cp.cuda.get_elapsed_time(start, end))

    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def benchmark_torch(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    warmup: int = 5,
    iterations: int = 20
) -> Dict[str, float]:
    """
    Benchmark a PyTorch function.

    Args:
        func: Function to benchmark
        args: Positional arguments
        kwargs: Keyword arguments
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        Dictionary with timing statistics
    """
    import torch

    kwargs = kwargs or {}

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = func(*args, **kwargs)
    torch.cuda.synchronize()

    # Timed iterations
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        start_events[i].record()
        with torch.no_grad():
            _ = func(*args, **kwargs)
        end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def print_benchmark_result(name: str, stats: Dict[str, float]):
    """Pretty print benchmark results."""
    print(f"{name}:")
    print(f"  Mean: {stats['mean_ms']:.3f} ms")
    print(f"  Min:  {stats['min_ms']:.3f} ms")
    print(f"  Max:  {stats['max_ms']:.3f} ms")
    if 'std_ms' in stats:
        print(f"  Std:  {stats['std_ms']:.3f} ms")


def compare_benchmarks(name1: str, stats1: Dict, name2: str, stats2: Dict):
    """Compare two benchmark results."""
    speedup = stats1['mean_ms'] / stats2['mean_ms']

    print(f"\n--- Comparison ---")
    print(f"{name1}: {stats1['mean_ms']:.3f} ms")
    print(f"{name2}: {stats2['mean_ms']:.3f} ms")

    if speedup > 1:
        print(f"Speedup: {speedup:.2f}x ({name2} is faster)")
    else:
        print(f"Slowdown: {1/speedup:.2f}x ({name1} is faster)")
