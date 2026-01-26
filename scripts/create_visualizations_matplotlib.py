#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Create performance visualizations for cutileGPT README using matplotlib.

This creates publication-quality static images.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11


class VisualizationCreator:
    """Create visualizations for cutileGPT project."""

    def __init__(self, output_dir="docs/assets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Actual benchmark data
        self.data = {
            'gelu_benchmark': {
                'tile_kernel_ms': 0.543,
                'cupy_kernel_ms': 4.314,
                'speedup': 7.95,
                'tensor_shape': '(32, 512, 768)',
                'total_elements': 12582912
            },
            'cutile_benchmarks': {
                'gpt_nano': {'latency_ms': 1.267, 'throughput': 202099},
                'gpt_small': {'latency_ms': 6.243, 'throughput': 41008},
                'gpt_medium': {'latency_ms': 10.132, 'throughput': 25267}
            },
            'pytorch_comparison': {
                'nano': {
                    'pytorch_ms': 2.372,
                    'cutile_ms': 2.888,
                    'ratio': 1.22
                },
                'medium': {
                    'pytorch_ms': 13.982,
                    'cutile_ms': 16.566,
                    'ratio': 1.18
                }
            }
        }

    def create_gelu_speedup_chart(self):
        """Create GELU kernel speedup visualization."""
        data = self.data['gelu_benchmark']

        fig, ax = plt.subplots(figsize=(10, 6))

        implementations = ['CuPy\nKernel', 'Tile\nKernel']
        times = [data['cupy_kernel_ms'], data['tile_kernel_ms']]
        colors = ['#ef4444', '#10b981']

        bars = ax.bar(implementations, times, color=colors, width=0.6, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.3f} ms',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')

        ax.set_ylabel('Latency (ms)', fontsize=14, fontweight='bold')
        ax.set_title(f'GELU Kernel: {data["speedup"]:.1f}x Speedup',
                    fontsize=16, fontweight='bold', pad=20)

        # Add subtitle
        fig.text(0.5, 0.92, f'Tensor: {data["tensor_shape"]} | {data["total_elements"]:,} elements',
                ha='center', fontsize=11, color='gray')

        # Add speedup annotation
        ax.annotate(f'{data["speedup"]:.1f}x\nFaster',
                   xy=(1, data['tile_kernel_ms']), xytext=(0.5, data['cupy_kernel_ms']*0.7),
                   arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                   fontsize=14, fontweight='bold', color='green',
                   ha='center')

        plt.tight_layout()
        return fig

    def create_cutile_performance_chart(self):
        """Create cutileGPT internal performance chart."""
        data = self.data['cutile_benchmarks']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        models = list(data.keys())
        latencies = [data[m]['latency_ms'] for m in models]
        throughputs = [data[m]['throughput'] / 1000 for m in models]  # Convert to K tok/s

        # Latency chart
        bars1 = ax1.bar(models, latencies, color='#3b82f6', width=0.6, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Latency (ms)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Model Configuration', fontsize=14, fontweight='bold')
        ax1.set_title('Latency', fontsize=14, fontweight='bold')

        for bar, lat in zip(bars1, latencies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{lat:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Throughput chart
        bars2 = ax2.bar(models, throughputs, color='#10b981', width=0.6, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Throughput (K tokens/sec)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Model Configuration', fontsize=14, fontweight='bold')
        ax2.set_title('Throughput', fontsize=14, fontweight='bold')

        for bar, thr in zip(bars2, throughputs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{thr:.0f}K',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        fig.suptitle('cutileGPT Performance (Batch=4, Seq=64)',
                    fontsize=16, fontweight='bold', y=1.0)

        plt.tight_layout()
        return fig

    def create_pytorch_comparison_chart(self):
        """Create PyTorch vs cutileGPT comparison chart."""
        data = self.data['pytorch_comparison']

        fig, ax = plt.subplots(figsize=(12, 7))

        models = ['Nano\n(3 layers)', 'Medium\n(6 layers)']
        pytorch_times = [data['nano']['pytorch_ms'], data['medium']['pytorch_ms']]
        cutile_times = [data['nano']['cutile_ms'], data['medium']['cutile_ms']]

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(x - width/2, pytorch_times, width, label='PyTorch minGPT',
                      color='#f97316', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, cutile_times, width, label='cutileGPT',
                      color='#3b82f6', edgecolor='black', linewidth=1.5)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Latency (ms)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model Configuration', fontsize=14, fontweight='bold')
        ax.set_title('PyTorch minGPT vs cutileGPT', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(fontsize=12, loc='upper left')

        # Add subtitle
        fig.text(0.5, 0.94, 'Batch=8, Seq=128 | Lower is better',
                ha='center', fontsize=11, color='gray')

        plt.tight_layout()
        return fig

    def create_tile_philosophy_comparison(self):
        """Create Tile Programming Philosophy comparison."""
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('off')

        # Table data
        aspects = ['Thread Management', 'Synchronization', 'Code Lines', 'Portability']
        traditional = ['Manual threadIdx.x', 'Explicit __syncthreads()', '~150 lines/kernel', 'GPU-specific']
        tile = ['Compiler handles', 'Automatic', '~20 lines/kernel', 'Hardware portable']
        benefits = ['87% less code', 'Zero bugs', 'Easy to read', 'Future-proof']

        # Create table
        table_data = []
        for i, aspect in enumerate(aspects):
            table_data.append([aspect, traditional[i], tile[i], benefits[i]])

        table = ax.table(cellText=table_data,
                        colLabels=['Aspect', 'Traditional CUDA', 'Tile Programming', 'Benefit'],
                        cellLoc='left',
                        loc='center',
                        bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # Style header
        for i in range(4):
            cell = table[(0, i)]
            cell.set_facecolor('#667eea')
            cell.set_text_props(weight='bold', color='white', fontsize=12)

        # Style cells
        for i in range(1, len(aspects) + 1):
            table[(i, 0)].set_facecolor('#f0f0f0')
            table[(i, 0)].set_text_props(weight='bold')
            table[(i, 1)].set_facecolor('#ffebee')
            table[(i, 2)].set_facecolor('#e8f5e9')
            table[(i, 3)].set_facecolor('#fff3cd')
            table[(i, 3)].set_text_props(weight='bold', color='#856404')

        plt.title('Traditional CUDA vs Tile Programming',
                 fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig

    def create_all_visualizations(self):
        """Create all visualizations and export as PNG."""
        print("Creating visualizations with matplotlib...")

        charts = {
            'gelu_speedup': self.create_gelu_speedup_chart(),
            'cutile_performance': self.create_cutile_performance_chart(),
            'pytorch_comparison': self.create_pytorch_comparison_chart(),
            'tile_philosophy': self.create_tile_philosophy_comparison()
        }

        for name, fig in charts.items():
            # Export as PNG (high DPI)
            png_path = self.output_dir / f"{name}.png"
            fig.savefig(str(png_path), dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  ✅ {png_path}")
            plt.close(fig)

        # Save data as JSON
        data_path = self.output_dir / "benchmark_data.json"
        with open(data_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"\n  💾 Data saved to {data_path}")

        print("\n✅ All visualizations created!")


def main():
    """Main entry point."""
    creator = VisualizationCreator()
    creator.create_all_visualizations()


if __name__ == "__main__":
    main()
