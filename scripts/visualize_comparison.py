#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
AS-IS vs TO-BE Comparison Visualization

Creates interactive visualizations comparing PyTorch minGPT (AS-IS)
with cutileGPT (TO-BE) using Plotly.

Usage:
    # Run comparison and visualize
    uv run python visualize_comparison.py --model tile-medium --benchmark

    # Compare multiple models
    uv run python visualize_comparison.py --compare-all --benchmark
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import torch
import cupy as cp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add minGPT to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external', 'minGPT'))
from mingpt.model import GPT

from cutile_gpt.model import CutileGPT, CutileGPTConfig


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
    gpt_config.embd_pdrop = 0.0
    gpt_config.resid_pdrop = 0.0
    gpt_config.attn_pdrop = 0.0

    model = GPT(gpt_config)
    model.eval()
    return model


def benchmark_forward_torch(model, idx, warmup=5, iterations=20):
    """Benchmark forward pass latency for PyTorch models."""
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(idx)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        start_events[i].record()
        with torch.no_grad():
            _ = model(idx)
        end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return {
        'mean_ms': sum(times) / len(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        'min_ms': min(times),
        'max_ms': max(times),
    }


def benchmark_forward_cupy(model, idx_cupy, warmup=5, iterations=20):
    """Benchmark forward pass latency for CuPy-based models."""
    for _ in range(warmup):
        _ = model.forward(idx_cupy)
    cp.cuda.Device().synchronize()

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
        'std_ms': float(cp.std(cp.array(times))),
        'min_ms': min(times),
        'max_ms': max(times),
    }


class ComparisonVisualizer:
    """Visualize AS-IS vs TO-BE comparisons."""

    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.data: Dict[str, Any] = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
            },
            'comparisons': {},
        }

    def run_comparison(self, model_name: str, batch_size: int = 4, seq_len: int = 64,
                      iterations: int = 50, use_fused_mlp: bool = True):
        """Run AS-IS vs TO-BE comparison for a model."""
        config_map = {
            'nano': CutileGPTConfig.gpt_nano,
            'micro': CutileGPTConfig.gpt_micro,
            'mini': CutileGPTConfig.gpt_mini,
            'tile-small': CutileGPTConfig.gpt_tile_small,
            'tile-medium': CutileGPTConfig.gpt_tile_medium,
            'tile-large': CutileGPTConfig.gpt_tile_large,
        }

        config = config_map[model_name]()
        config.vocab_size = 50257
        config.block_size = max(config.block_size, seq_len)

        print(f"\n{'='*80}")
        print(f"Comparing: {model_name}")
        print(f"{'='*80}")
        print(f"Config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dims")
        print(f"Batch: {batch_size}, Seq: {seq_len}, Iterations: {iterations}")

        # Create models
        print("\nCreating minGPT (AS-IS)...")
        mingpt_model = create_mingpt_model(config).cuda()

        print(f"Creating cutileGPT (TO-BE, fused_mlp={use_fused_mlp})...")
        cutile_model = CutileGPT(config, device='cuda', use_fused_mlp=use_fused_mlp)

        # Load weights
        cutile_model.load_from_mingpt(mingpt_model)

        # Create input
        idx_torch = torch.randint(0, config.vocab_size, (batch_size, seq_len), device='cuda')
        idx_cupy = cp.asarray(idx_torch.cpu().numpy())

        # Benchmark AS-IS
        print("\nBenchmarking minGPT (AS-IS)...")
        mingpt_stats = benchmark_forward_torch(mingpt_model, idx_torch, iterations=iterations)
        print(f"  Mean: {mingpt_stats['mean_ms']:.3f} ms")

        # Benchmark TO-BE
        print("Benchmarking cutileGPT (TO-BE)...")
        cutile_stats = benchmark_forward_cupy(cutile_model, idx_cupy, iterations=iterations)
        print(f"  Mean: {cutile_stats['mean_ms']:.3f} ms")

        # Calculate speedup
        speedup = mingpt_stats['mean_ms'] / cutile_stats['mean_ms']
        print(f"\nSpeedup: {speedup:.2f}x")

        # Store results
        self.data['comparisons'][model_name] = {
            'config': {
                'n_layer': config.n_layer,
                'n_head': config.n_head,
                'n_embd': config.n_embd,
            },
            'mingpt': mingpt_stats,
            'cutile': cutile_stats,
            'speedup': speedup,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'fused_mlp': use_fused_mlp,
        }

        return self.data['comparisons'][model_name]

    def create_speedup_chart(self):
        """Create speedup comparison chart."""
        models = list(self.data['comparisons'].keys())
        speedups = [self.data['comparisons'][m]['speedup'] for m in models]

        # Color based on speedup
        colors = ['green' if s > 1.0 else 'red' for s in speedups]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=models,
            y=speedups,
            marker_color=colors,
            text=[f"{s:.2f}x" for s in speedups],
            textposition='outside',
            name='Speedup',
        ))

        # Add reference line at 1.0x
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                     annotation_text="Baseline (1.0x)")

        fig.update_layout(
            title='AS-IS (minGPT) vs TO-BE (cutileGPT) Speedup',
            xaxis_title='Model Configuration',
            yaxis_title='Speedup Factor',
            template='plotly_white',
            height=500,
        )

        return fig

    def create_latency_comparison_chart(self):
        """Create side-by-side latency comparison."""
        models = list(self.data['comparisons'].keys())
        mingpt_latencies = [self.data['comparisons'][m]['mingpt']['mean_ms'] for m in models]
        cutile_latencies = [self.data['comparisons'][m]['cutile']['mean_ms'] for m in models]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='AS-IS (minGPT)',
            x=models,
            y=mingpt_latencies,
            marker_color='rgb(55, 83, 109)',
        ))

        fig.add_trace(go.Bar(
            name='TO-BE (cutileGPT)',
            x=models,
            y=cutile_latencies,
            marker_color='rgb(26, 118, 255)',
        ))

        fig.update_layout(
            title='Forward Pass Latency: AS-IS vs TO-BE',
            xaxis_title='Model Configuration',
            yaxis_title='Latency (ms)',
            barmode='group',
            template='plotly_white',
            height=500,
        )

        return fig

    def create_detailed_comparison_chart(self):
        """Create detailed comparison with error bars."""
        models = list(self.data['comparisons'].keys())

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('AS-IS (minGPT)', 'TO-BE (cutileGPT)'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )

        # AS-IS data
        mingpt_means = [self.data['comparisons'][m]['mingpt']['mean_ms'] for m in models]
        mingpt_stds = [self.data['comparisons'][m]['mingpt']['std_ms'] for m in models]

        # TO-BE data
        cutile_means = [self.data['comparisons'][m]['cutile']['mean_ms'] for m in models]
        cutile_stds = [self.data['comparisons'][m]['cutile']['std_ms'] for m in models]

        fig.add_trace(
            go.Bar(
                x=models,
                y=mingpt_means,
                error_y=dict(type='data', array=mingpt_stds),
                marker_color='rgb(55, 83, 109)',
                name='AS-IS',
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=models,
                y=cutile_means,
                error_y=dict(type='data', array=cutile_stds),
                marker_color='rgb(26, 118, 255)',
                name='TO-BE',
            ),
            row=1, col=2
        )

        fig.update_layout(
            title_text='Detailed Latency Comparison with Standard Deviation',
            template='plotly_white',
            height=500,
            showlegend=False,
        )

        return fig

    def create_dashboard(self, output_file: str = "comparison_dashboard.html"):
        """Create comparison dashboard."""
        print("\nGenerating comparison dashboard...")

        charts = [
            self.create_speedup_chart(),
            self.create_latency_comparison_chart(),
            self.create_detailed_comparison_chart(),
        ]

        output_path = self.output_dir / output_file

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AS-IS vs TO-BE Comparison Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
        }}
        .speedup-positive {{
            color: green;
            font-weight: bold;
        }}
        .speedup-negative {{
            color: red;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔄 AS-IS vs TO-BE Comparison</h1>
        <p>PyTorch minGPT (AS-IS) vs cutileGPT (TO-BE)</p>
        <p>Timestamp: {self.data['metadata']['timestamp']}</p>
    </div>

    <div class="summary">
        <h2>Comparison Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>AS-IS (ms)</th>
                <th>TO-BE (ms)</th>
                <th>Speedup</th>
            </tr>
"""

        for model, data in self.data['comparisons'].items():
            speedup_class = 'speedup-positive' if data['speedup'] > 1.0 else 'speedup-negative'
            html_content += f"""
            <tr>
                <td>{model}</td>
                <td>{data['mingpt']['mean_ms']:.3f}</td>
                <td>{data['cutile']['mean_ms']:.3f}</td>
                <td class="{speedup_class}">{data['speedup']:.2f}x</td>
            </tr>
"""

        html_content += f"""
        </table>
    </div>

    <div class="chart-container" id="chart1"></div>
    <div class="chart-container" id="chart2"></div>
    <div class="chart-container" id="chart3"></div>

    <script>
        var chart1 = {charts[0].to_json()};
        var chart2 = {charts[1].to_json()};
        var chart3 = {charts[2].to_json()};

        Plotly.newPlot('chart1', chart1.data, chart1.layout, {{responsive: true}});
        Plotly.newPlot('chart2', chart2.data, chart2.layout, {{responsive: true}});
        Plotly.newPlot('chart3', chart3.data, chart3.layout, {{responsive: true}});
    </script>
</body>
</html>
"""

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"Dashboard saved to: {output_path}")
        return output_path

    def save_data(self, output_file: str = "comparison_data.json"):
        """Save comparison data to JSON."""
        output_path = self.output_dir / output_file

        with open(output_path, 'w') as f:
            json.dump(self.data, f, indent=2)

        print(f"Data saved to: {output_path}")
        return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AS-IS vs TO-BE comparison visualization')
    parser.add_argument('--model', type=str, default='tile-medium',
                       choices=['nano', 'micro', 'mini', 'tile-small', 'tile-medium', 'tile-large'],
                       help='Model to compare')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all model configurations')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmarks (required for comparison)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--seq-len', type=int, default=64,
                       help='Sequence length')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Benchmark iterations')
    parser.add_argument('--fused-mlp', action='store_true', default=True,
                       help='Use fused MLP kernel')
    parser.add_argument('--output-dir', type=str, default='profiling_results',
                       help='Output directory')

    args = parser.parse_args()

    if not args.benchmark:
        print("Error: --benchmark flag required to run comparison")
        print("Usage: python visualize_comparison.py --model tile-medium --benchmark")
        return

    visualizer = ComparisonVisualizer(output_dir=args.output_dir)

    if args.compare_all:
        models = ['nano', 'micro', 'mini', 'tile-small', 'tile-medium', 'tile-large']
        for model in models:
            visualizer.run_comparison(
                model,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                iterations=args.iterations,
                use_fused_mlp=args.fused_mlp
            )
    else:
        visualizer.run_comparison(
            args.model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            iterations=args.iterations,
            use_fused_mlp=args.fused_mlp
        )

    # Save data and create dashboard
    visualizer.save_data()
    dashboard_path = visualizer.create_dashboard()

    print("\n" + "=" * 80)
    print("✅ Comparison Complete!")
    print("=" * 80)
    print(f"\nOpen the dashboard:")
    print(f"  file://{dashboard_path.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
