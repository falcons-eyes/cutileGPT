#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
AS-IS vs TO-BE Comparison Script

Compares PyTorch minGPT (AS-IS) with cutile GPT (TO-BE).
"""

import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import cupy as cp

# Add minGPT to path (external submodule)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'minGPT'))
from mingpt.model import GPT

from model import CutileGPT, CutileGPTConfig


def create_mingpt_model(config: CutileGPTConfig) -> GPT:
    """Create a minGPT model with matching configuration."""
    from mingpt.utils import CfgNode as CN

    gpt_config = GPT.get_default_config()
    gpt_config.model_type = None
    gpt_config.n_layer = config.n_layer
    gpt_config.n_head = config.n_head
    gpt_config.n_embd = config.n_embd
    gpt_config.vocab_size = config.vocab_size
    gpt_config.block_size = config.block_size
    gpt_config.embd_pdrop = 0.0  # No dropout for inference
    gpt_config.resid_pdrop = 0.0
    gpt_config.attn_pdrop = 0.0

    model = GPT(gpt_config)
    model.eval()
    return model


def benchmark_forward_torch(model, idx, warmup=5, iterations=20):
    """Benchmark forward pass latency for PyTorch models."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(idx) if hasattr(model, 'forward') else model.forward(idx)
    torch.cuda.synchronize()

    # Timed iterations
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        start_events[i].record()
        with torch.no_grad():
            _ = model(idx) if hasattr(model, 'forward') else model.forward(idx)
        end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
    }


def benchmark_forward_cupy(model, idx_cupy, warmup=5, iterations=20):
    """Benchmark forward pass latency for CuPy-based models."""
    # Warmup
    for _ in range(warmup):
        _ = model.forward(idx_cupy)
    cp.cuda.Device().synchronize()

    # Timed iterations
    times = []
    for i in range(iterations):
        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        _ = model.forward(idx_cupy)
        end.record()
        end.synchronize()

        times.append(cp.cuda.get_elapsed_time(start, end))

    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
    }


def compare_outputs(mingpt_model, cutile_model, idx_torch, atol=1e-3, rtol=1e-3):
    """Compare outputs between minGPT and cutile GPT."""
    with torch.no_grad():
        # minGPT forward
        mingpt_logits, _ = mingpt_model(idx_torch)

        # Convert to cupy for cutile GPT
        idx_cupy = cp.asarray(idx_torch.cpu().numpy())

        # cutile GPT forward
        cutile_logits_cupy, _ = cutile_model.forward(idx_cupy)

        # Convert back to torch for comparison
        cutile_logits = torch.from_numpy(cp.asnumpy(cutile_logits_cupy)).cuda()

    # Compare
    diff = (mingpt_logits - cutile_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Check if close
    is_close = torch.allclose(mingpt_logits, cutile_logits, atol=atol, rtol=rtol)

    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'is_close': is_close,
        'mingpt_logits': mingpt_logits,
        'cutile_logits': cutile_logits,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare minGPT vs cutile GPT')
    parser.add_argument('--model', type=str, default='nano',
                        choices=['nano', 'micro', 'mini', 'tile-small', 'tile-medium', 'tile-large'],
                        help='Model size (default: nano). tile-* uses power of 2 dims')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--seq-len', type=int, default=32,
                        help='Sequence length (default: 32)')
    parser.add_argument('--vocab-size', type=int, default=100,
                        help='Vocabulary size (default: 100)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark')
    parser.add_argument('--iterations', type=int, default=20,
                        help='Benchmark iterations (default: 20)')
    parser.add_argument('--fused-mlp', action='store_true',
                        help='Use fused MLP kernel (single kernel for Linear->GELU->Linear)')
    args = parser.parse_args()

    print("=" * 60)
    print("AS-IS (PyTorch minGPT) vs TO-BE (cutile GPT) Comparison")
    print("=" * 60)

    # Select config
    config_map = {
        'nano': CutileGPTConfig.gpt_nano,
        'micro': CutileGPTConfig.gpt_micro,
        'mini': CutileGPTConfig.gpt_mini,
        'tile-small': CutileGPTConfig.gpt_tile_small,
        'tile-medium': CutileGPTConfig.gpt_tile_medium,
        'tile-large': CutileGPTConfig.gpt_tile_large,
    }
    config = config_map[args.model]()
    config.vocab_size = args.vocab_size
    config.block_size = max(config.block_size, args.seq_len)

    is_tile_optimized = args.model.startswith('tile-')

    print(f"\nModel Configuration:")
    print(f"  Model: gpt-{args.model}")
    print(f"  Tile-optimized (power of 2): {is_tile_optimized}")
    print(f"  Layers: {config.n_layer}")
    print(f"  Heads: {config.n_head}")
    print(f"  Embedding dim: {config.n_embd}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Block size: {config.block_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Fused MLP: {args.fused_mlp}")

    # Create models
    print("\n--- Creating Models ---")

    print("Creating minGPT model (AS-IS)...")
    mingpt_model = create_mingpt_model(config).cuda()

    print(f"Creating cutile GPT model (TO-BE, fused_mlp={args.fused_mlp})...")
    cutile_model = CutileGPT(config, device='cuda', use_fused_mlp=args.fused_mlp)

    # Load weights from minGPT to cutile
    print("Loading weights from minGPT to cutile GPT...")
    cutile_model.load_from_mingpt(mingpt_model)

    # Create test input
    idx_torch = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device='cuda')

    # Compare outputs
    print("\n--- Correctness Comparison ---")
    comparison = compare_outputs(mingpt_model, cutile_model, idx_torch)

    print(f"Max absolute difference: {comparison['max_diff']:.6e}")
    print(f"Mean absolute difference: {comparison['mean_diff']:.6e}")

    if comparison['is_close']:
        print("\n[PASS] Outputs match within tolerance!")
    else:
        print("\n[FAIL] Outputs differ beyond tolerance")
        print("Note: Some difference is expected due to numerical precision")

    # Check top-k predictions match
    mingpt_topk = comparison['mingpt_logits'][:, -1, :].topk(5).indices
    cutile_topk = comparison['cutile_logits'][:, -1, :].topk(5).indices

    top1_match = (mingpt_topk[:, 0] == cutile_topk[:, 0]).all().item()
    top5_match = (mingpt_topk == cutile_topk).all().item()

    print(f"\nTop-1 prediction match: {top1_match}")
    print(f"Top-5 prediction match: {top5_match}")

    # Performance benchmark
    if args.benchmark:
        print("\n--- Performance Benchmark ---")
        print(f"Warmup: 5 iterations, Timed: {args.iterations} iterations")

        # minGPT benchmark
        print("\nBenchmarking minGPT (AS-IS)...")
        mingpt_stats = benchmark_forward_torch(mingpt_model, idx_torch, iterations=args.iterations)
        print(f"  Mean: {mingpt_stats['mean_ms']:.3f} ms")
        print(f"  Min:  {mingpt_stats['min_ms']:.3f} ms")
        print(f"  Max:  {mingpt_stats['max_ms']:.3f} ms")

        # cutile GPT benchmark
        idx_cupy = cp.asarray(idx_torch.cpu().numpy())
        print("\nBenchmarking cutile GPT (TO-BE)...")
        cutile_stats = benchmark_forward_cupy(cutile_model, idx_cupy, iterations=args.iterations)
        print(f"  Mean: {cutile_stats['mean_ms']:.3f} ms")
        print(f"  Min:  {cutile_stats['min_ms']:.3f} ms")
        print(f"  Max:  {cutile_stats['max_ms']:.3f} ms")

        # Comparison
        speedup = mingpt_stats['mean_ms'] / cutile_stats['mean_ms']
        print(f"\n--- Performance Summary ---")
        print(f"minGPT (AS-IS):  {mingpt_stats['mean_ms']:.3f} ms")
        print(f"cutile (TO-BE):  {cutile_stats['mean_ms']:.3f} ms")
        if speedup > 1:
            print(f"Speedup: {speedup:.2f}x faster")
        else:
            print(f"Slowdown: {1/speedup:.2f}x slower")

    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
