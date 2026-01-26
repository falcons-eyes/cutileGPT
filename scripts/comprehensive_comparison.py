#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive PyTorch vs cutileGPT comparison across multiple dimensions.

Compares across:
- Model sizes: nano, small, medium
- Batch sizes: 1, 4, 8, 16
- Sequence lengths: 64, 128, 256
- Metrics: latency, throughput, memory, speedup
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'external', 'minGPT'))

import torch
import cupy as cp
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from mingpt.model import GPT
from cutile_gpt.model_tile import CutileGPT, GPTConfig


class ComprehensiveComparison:
    """Multi-dimensional PyTorch vs cutileGPT comparison."""

    def __init__(self, output_dir="docs/assets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test configurations
        self.model_configs = {
            'nano': GPTConfig(n_layer=3, n_head=3, n_embd=48),
            'small': GPTConfig(n_layer=6, n_head=6, n_embd=384),
            'medium': GPTConfig(n_layer=8, n_head=8, n_embd=512),
        }

        self.batch_sizes = [1, 4, 8, 16]
        self.seq_lengths = [64, 128, 256]

        self.results = []

    def create_pytorch_model(self, config: GPTConfig) -> GPT:
        """Create PyTorch minGPT model."""
        gpt_config = GPT.get_default_config()
        gpt_config.model_type = None
        gpt_config.n_layer = config.n_layer
        gpt_config.n_head = config.n_head
        gpt_config.n_embd = config.n_embd
        gpt_config.vocab_size = 50257
        gpt_config.block_size = 512
        gpt_config.embd_pdrop = 0.0
        gpt_config.resid_pdrop = 0.0
        gpt_config.attn_pdrop = 0.0

        model = GPT(gpt_config).cuda().eval()
        return model

    def benchmark_pytorch(self, model, tokens, iterations=30):
        """Benchmark PyTorch model."""
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(tokens)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            with torch.no_grad():
                _ = model(tokens)
            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)

        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }

    def benchmark_cutile(self, model, tokens, iterations=30):
        """Benchmark cutileGPT model."""
        # Warmup
        for _ in range(5):
            _ = model.forward(tokens)
        cp.cuda.Stream.null.synchronize()

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            _ = model.forward(tokens)
            cp.cuda.Stream.null.synchronize()
            times.append((time.time() - start) * 1000)

        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }

    def run_comparison(self, model_name: str, batch_size: int, seq_len: int):
        """Run single comparison."""
        config = self.model_configs[model_name]

        print(f"\n{'='*70}")
        print(f"Model: {model_name} | Batch: {batch_size} | Seq: {seq_len}")
        print(f"{'='*70}")

        # Create models
        print("Creating models...")
        pytorch_model = self.create_pytorch_model(config)
        cutile_model = CutileGPT(config)

        # Load weights
        cutile_model.load_from_minGPT(pytorch_model.state_dict())

        # Create inputs
        tokens_torch = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
        tokens_cupy = cp.array(tokens_torch.cpu().numpy(), dtype=cp.int32)

        # Benchmark PyTorch
        print("Benchmarking PyTorch...")
        pytorch_stats = self.benchmark_pytorch(pytorch_model, tokens_torch)

        # Benchmark cutileGPT
        print("Benchmarking cutileGPT...")
        cutile_stats = self.benchmark_cutile(cutile_model, tokens_cupy)

        # Calculate metrics
        total_tokens = batch_size * seq_len
        pytorch_throughput = total_tokens / (pytorch_stats['mean'] / 1000)
        cutile_throughput = total_tokens / (cutile_stats['mean'] / 1000)

        speedup = pytorch_stats['mean'] / cutile_stats['mean']

        result = {
            'model': model_name,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'total_tokens': total_tokens,
            'pytorch_latency_ms': pytorch_stats['mean'],
            'cutile_latency_ms': cutile_stats['mean'],
            'pytorch_throughput': pytorch_throughput,
            'cutile_throughput': cutile_throughput,
            'speedup': speedup,
            'pytorch_std': pytorch_stats['std'],
            'cutile_std': cutile_stats['std'],
        }

        print(f"PyTorch:  {pytorch_stats['mean']:.3f} ms ({pytorch_throughput:,.0f} tok/s)")
        print(f"cutileGPT: {cutile_stats['mean']:.3f} ms ({cutile_throughput:,.0f} tok/s)")
        print(f"Ratio: {speedup:.3f}x")

        self.results.append(result)

        # Clean up
        del pytorch_model, cutile_model, tokens_torch, tokens_cupy
        torch.cuda.empty_cache()

        return result

    def run_all_comparisons(self):
        """Run all comparison combinations."""
        print("="*70)
        print("Comprehensive PyTorch vs cutileGPT Comparison")
        print("="*70)

        total = len(self.model_configs) * len(self.batch_sizes) * len(self.seq_lengths)
        current = 0

        for model_name in self.model_configs.keys():
            for batch_size in self.batch_sizes:
                for seq_len in self.seq_lengths:
                    current += 1
                    print(f"\nProgress: {current}/{total}")
                    self.run_comparison(model_name, batch_size, seq_len)

        print("\n" + "="*70)
        print("All comparisons complete!")
        print("="*70)

    def save_results(self):
        """Save results to JSON and CSV."""
        # JSON
        json_path = self.output_dir / "comprehensive_comparison.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to {json_path}")

        # CSV
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "comprehensive_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 CSV saved to {csv_path}")

        return df

    def create_summary_table(self, df: pd.DataFrame):
        """Create summary comparison table."""
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')

        # Prepare table data
        table_data = []
        headers = ['Model', 'Batch', 'Seq', 'PyTorch\n(ms)', 'cutileGPT\n(ms)',
                   'PyTorch\n(tok/s)', 'cutileGPT\n(tok/s)', 'Ratio']

        for _, row in df.iterrows():
            table_data.append([
                row['model'],
                row['batch_size'],
                row['seq_len'],
                f"{row['pytorch_latency_ms']:.2f}",
                f"{row['cutile_latency_ms']:.2f}",
                f"{row['pytorch_throughput']/1000:.1f}K",
                f"{row['cutile_throughput']/1000:.1f}K",
                f"{row['speedup']:.3f}x"
            ])

        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#4f46e5')
            cell.set_text_props(weight='bold', color='white')

        # Color rows by speedup
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            speedup = row['speedup']
            if speedup >= 1.0:
                color = '#d1fae5'  # Green
            elif speedup >= 0.9:
                color = '#fef3c7'  # Yellow
            else:
                color = '#fee2e2'  # Red

            for j in range(len(headers)):
                table[(i, j)].set_facecolor(color)

        plt.title('Comprehensive PyTorch vs cutileGPT Comparison',
                 fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        path = self.output_dir / "comparison_table.png"
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✅ {path}")
        plt.close(fig)

    def create_heatmaps(self, df: pd.DataFrame):
        """Create heatmaps for different metrics."""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))

        for model_idx, model_name in enumerate(self.model_configs.keys()):
            model_df = df[df['model'] == model_name]

            # Latency comparison
            ax = axes[model_idx, 0]
            pivot = model_df.pivot(index='seq_len', columns='batch_size', values='pytorch_latency_ms')
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'ms'})
            ax.set_title(f'{model_name.capitalize()} - PyTorch Latency', fontweight='bold')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Sequence Length')

            # Speedup ratio
            ax = axes[model_idx, 1]
            pivot = model_df.pivot(index='seq_len', columns='batch_size', values='speedup')
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0, ax=ax,
                       cbar_kws={'label': 'Ratio'})
            ax.set_title(f'{model_name.capitalize()} - Performance Ratio', fontweight='bold')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Sequence Length')

        plt.suptitle('Performance Heatmaps: PyTorch vs cutileGPT',
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()

        path = self.output_dir / "comparison_heatmaps.png"
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✅ {path}")
        plt.close(fig)

    def create_throughput_comparison(self, df: pd.DataFrame):
        """Create throughput comparison chart."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, model_name in enumerate(self.model_configs.keys()):
            ax = axes[idx]
            model_df = df[df['model'] == model_name]

            # Group by batch size
            for batch_size in self.batch_sizes:
                batch_df = model_df[model_df['batch_size'] == batch_size]

                ax.plot(batch_df['seq_len'], batch_df['pytorch_throughput']/1000,
                       marker='o', label=f'PyTorch B={batch_size}', linestyle='--')
                ax.plot(batch_df['seq_len'], batch_df['cutile_throughput']/1000,
                       marker='s', label=f'cutileGPT B={batch_size}')

            ax.set_title(f'{model_name.capitalize()} Model', fontweight='bold', fontsize=14)
            ax.set_xlabel('Sequence Length', fontweight='bold')
            ax.set_ylabel('Throughput (K tokens/sec)', fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Throughput Comparison Across Configurations',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        path = self.output_dir / "throughput_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✅ {path}")
        plt.close(fig)

    def create_visualizations(self, df: pd.DataFrame):
        """Create all visualizations."""
        print("\nCreating visualizations...")

        self.create_summary_table(df)
        self.create_heatmaps(df)
        self.create_throughput_comparison(df)

        print("\n✅ All visualizations created!")

    def generate_markdown_table(self, df: pd.DataFrame):
        """Generate markdown table for README."""
        md_path = self.output_dir / "comparison_table.md"

        with open(md_path, 'w') as f:
            f.write("# PyTorch vs cutileGPT Comprehensive Comparison\n\n")

            for model_name in self.model_configs.keys():
                model_df = df[df['model'] == model_name]

                f.write(f"## {model_name.capitalize()} Model\n\n")
                f.write("| Batch | Seq | PyTorch (ms) | cutileGPT (ms) | PyTorch (tok/s) | cutileGPT (tok/s) | Ratio |\n")
                f.write("|-------|-----|--------------|----------------|-----------------|-------------------|-------|\n")

                for _, row in model_df.iterrows():
                    f.write(f"| {row['batch_size']} | {row['seq_len']} | "
                           f"{row['pytorch_latency_ms']:.2f} | {row['cutile_latency_ms']:.2f} | "
                           f"{row['pytorch_throughput']:,.0f} | {row['cutile_throughput']:,.0f} | "
                           f"{row['speedup']:.3f}x |\n")

                f.write("\n")

        print(f"  ✅ Markdown table: {md_path}")


def main():
    """Main entry point."""
    comparison = ComprehensiveComparison()

    # Run all comparisons
    comparison.run_all_comparisons()

    # Save results
    df = comparison.save_results()

    # Create visualizations
    comparison.create_visualizations(df)

    # Generate markdown table
    comparison.generate_markdown_table(df)

    print("\n" + "="*70)
    print("✅ Comprehensive comparison complete!")
    print("="*70)
    print(f"\nResults saved to: {comparison.output_dir}")
    print("  - comprehensive_comparison.json")
    print("  - comprehensive_comparison.csv")
    print("  - comparison_table.png")
    print("  - comparison_heatmaps.png")
    print("  - throughput_comparison.png")
    print("  - comparison_table.md")


if __name__ == "__main__":
    main()
