#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Performance Visualization Dashboard

Creates interactive visualizations for cutileGPT performance profiling data.
Uses Plotly for modern, interactive charts and exports to standalone HTML.

Usage:
    # Run profiling and generate visualizations
    python visualize_performance.py --run-profiling

    # Just create visualizations from existing data
    python visualize_performance.py --data profiling_data.json

    # Generate visualizations with custom output
    python visualize_performance.py --output custom_dashboard.html
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import cupy as cp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cutile_gpt.model import CutileGPT, CutileGPTConfig


class PerformanceDashboard:
    """Interactive performance dashboard using Plotly."""

    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.data: Dict[str, Any] = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'cupy_version': cp.__version__,
                'cuda_version': str(cp.cuda.runtime.runtimeGetVersion()),
            },
            'benchmarks': {},
        }

        # Get GPU info
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        self.data['metadata']['gpu_name'] = props['name'].decode()
        self.data['metadata']['compute_capability'] = f"{props['major']}.{props['minor']}"

    def benchmark_forward_pass(self, model, idx, warmup=10, iterations=100):
        """Benchmark forward pass with GPU timing."""
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

            elapsed = cp.cuda.get_elapsed_time(start, end)
            times.append(elapsed)

        times = cp.array(times)
        return {
            'mean': float(cp.mean(times)),
            'std': float(cp.std(times)),
            'min': float(cp.min(times)),
            'max': float(cp.max(times)),
            'median': float(cp.median(times)),
            'p95': float(cp.percentile(times, 95)),
            'p99': float(cp.percentile(times, 99)),
        }

    def run_profiling(self, batch_size=4, seq_len=64):
        """Run comprehensive profiling across all model configs."""
        configs = {
            'gpt_nano': CutileGPTConfig.gpt_nano(),
            'gpt_micro': CutileGPTConfig.gpt_micro(),
            'gpt_mini': CutileGPTConfig.gpt_mini(),
            'gpt_tile_small': CutileGPTConfig.gpt_tile_small(),
            'gpt_tile_medium': CutileGPTConfig.gpt_tile_medium(),
            'gpt_tile_large': CutileGPTConfig.gpt_tile_large(),
        }

        print("=" * 80)
        print("Running Performance Profiling")
        print("=" * 80)
        print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
        print(f"GPU: {self.data['metadata']['gpu_name']}")
        print("=" * 80)

        for name, config in configs.items():
            print(f"\nProfiling: {name}")
            print(f"  Config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dims")

            # Create model (fused_mlp=False due to performance issues)
            model = CutileGPT(config, use_fused_mlp=False)

            # Create input
            idx = cp.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=cp.int64)

            # Benchmark
            stats = self.benchmark_forward_pass(model, idx, warmup=10, iterations=100)

            # Calculate throughput
            throughput = (batch_size * seq_len * 1000) / stats['mean']

            # Store results
            self.data['benchmarks'][name] = {
                'config': {
                    'n_layer': config.n_layer,
                    'n_head': config.n_head,
                    'n_embd': config.n_embd,
                    'vocab_size': config.vocab_size,
                },
                'timing': stats,
                'throughput_tokens_per_sec': throughput,
                'batch_size': batch_size,
                'seq_len': seq_len,
            }

            print(f"  Mean: {stats['mean']:.4f} ms")
            print(f"  Throughput: {throughput:.0f} tokens/sec")

        print("\n" + "=" * 80)
        print("Profiling Complete!")
        print("=" * 80)

        return self.data

    def create_latency_chart(self):
        """Create interactive latency comparison chart."""
        configs = list(self.data['benchmarks'].keys())
        means = [self.data['benchmarks'][c]['timing']['mean'] for c in configs]
        stds = [self.data['benchmarks'][c]['timing']['std'] for c in configs]
        mins = [self.data['benchmarks'][c]['timing']['min'] for c in configs]
        maxs = [self.data['benchmarks'][c]['timing']['max'] for c in configs]

        fig = go.Figure()

        # Add bar chart with error bars
        fig.add_trace(go.Bar(
            x=configs,
            y=means,
            error_y=dict(type='data', array=stds),
            name='Mean Latency',
            marker_color='rgb(55, 83, 109)',
            text=[f"{m:.2f}ms" for m in means],
            textposition='outside',
        ))

        # Add min/max range as scatter
        fig.add_trace(go.Scatter(
            x=configs,
            y=mins,
            mode='markers',
            name='Min',
            marker=dict(color='green', size=8, symbol='triangle-down'),
        ))

        fig.add_trace(go.Scatter(
            x=configs,
            y=maxs,
            mode='markers',
            name='Max',
            marker=dict(color='red', size=8, symbol='triangle-up'),
        ))

        fig.update_layout(
            title='Forward Pass Latency Comparison',
            xaxis_title='Model Configuration',
            yaxis_title='Latency (ms)',
            template='plotly_white',
            height=500,
            showlegend=True,
            hovermode='x unified',
        )

        return fig

    def create_throughput_chart(self):
        """Create interactive throughput comparison chart."""
        configs = list(self.data['benchmarks'].keys())
        throughputs = [self.data['benchmarks'][c]['throughput_tokens_per_sec'] for c in configs]

        # Color gradient based on throughput
        colors = ['rgb(255, 0, 0)' if 'tile' in c else 'rgb(0, 0, 255)' for c in configs]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=configs,
            y=throughputs,
            marker_color=colors,
            text=[f"{t/1000:.1f}K" for t in throughputs],
            textposition='outside',
            name='Throughput',
        ))

        fig.update_layout(
            title='Throughput Comparison (tokens/sec)',
            xaxis_title='Model Configuration',
            yaxis_title='Tokens per Second',
            template='plotly_white',
            height=500,
            showlegend=False,
        )

        return fig

    def create_percentile_chart(self):
        """Create latency percentile distribution chart."""
        configs = list(self.data['benchmarks'].keys())

        fig = go.Figure()

        percentiles = ['min', 'median', 'p95', 'p99', 'max']
        colors = ['green', 'blue', 'orange', 'red', 'darkred']

        for i, p in enumerate(percentiles):
            values = [self.data['benchmarks'][c]['timing'][p] for c in configs]
            fig.add_trace(go.Scatter(
                x=configs,
                y=values,
                mode='lines+markers',
                name=p.upper(),
                line=dict(color=colors[i], width=2),
                marker=dict(size=8),
            ))

        fig.update_layout(
            title='Latency Percentile Distribution',
            xaxis_title='Model Configuration',
            yaxis_title='Latency (ms)',
            template='plotly_white',
            height=500,
            hovermode='x unified',
        )

        return fig

    def create_model_size_chart(self):
        """Create model size comparison chart."""
        configs = list(self.data['benchmarks'].keys())

        # Calculate approximate model sizes
        layers = [self.data['benchmarks'][c]['config']['n_layer'] for c in configs]
        embds = [self.data['benchmarks'][c]['config']['n_embd'] for c in configs]
        heads = [self.data['benchmarks'][c]['config']['n_head'] for c in configs]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Layers', 'Embedding Dim', 'Attention Heads'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
        )

        fig.add_trace(
            go.Bar(x=configs, y=layers, name='Layers', marker_color='lightblue'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=configs, y=embds, name='Embd Dim', marker_color='lightgreen'),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(x=configs, y=heads, name='Heads', marker_color='lightcoral'),
            row=1, col=3
        )

        fig.update_layout(
            title_text='Model Architecture Comparison',
            template='plotly_white',
            height=400,
            showlegend=False,
        )

        return fig

    def create_efficiency_chart(self):
        """Create efficiency chart (throughput per layer)."""
        configs = list(self.data['benchmarks'].keys())
        throughputs = [self.data['benchmarks'][c]['throughput_tokens_per_sec'] for c in configs]
        layers = [self.data['benchmarks'][c]['config']['n_layer'] for c in configs]
        embds = [self.data['benchmarks'][c]['config']['n_embd'] for c in configs]

        efficiency = [t / l for t, l in zip(throughputs, layers)]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=layers,
            y=efficiency,
            mode='markers+text',
            marker=dict(
                size=[e / 10 for e in embds],  # Size based on embedding dim
                color=throughputs,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Throughput<br>(tok/s)"),
            ),
            text=configs,
            textposition='top center',
            name='Configs',
        ))

        fig.update_layout(
            title='Model Efficiency (Throughput per Layer)',
            xaxis_title='Number of Layers',
            yaxis_title='Throughput per Layer (tokens/sec/layer)',
            template='plotly_white',
            height=500,
        )

        return fig

    def create_dashboard(self, output_file: str = "performance_dashboard.html"):
        """Create comprehensive performance dashboard."""
        print("\nGenerating interactive dashboard...")

        # Create all charts
        charts = [
            self.create_latency_chart(),
            self.create_throughput_chart(),
            self.create_percentile_chart(),
            self.create_model_size_chart(),
            self.create_efficiency_chart(),
        ]

        # Create multi-page HTML with all charts
        output_path = self.output_dir / output_file

        # Write HTML with custom layout
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>cutileGPT Performance Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
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
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .metadata {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metadata-item {{
            padding: 10px;
            background: #f9f9f9;
            border-radius: 5px;
        }}
        .metadata-label {{
            font-weight: bold;
            color: #667eea;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .metadata-value {{
            font-size: 16px;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 cutileGPT Performance Dashboard</h1>
        <p>Interactive visualization of NVIDIA cutile-optimized GPT performance metrics</p>
    </div>

    <div class="metadata">
        <h2>System Information</h2>
        <div class="metadata-grid">
            <div class="metadata-item">
                <div class="metadata-label">GPU</div>
                <div class="metadata-value">{self.data['metadata']['gpu_name']}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Compute Capability</div>
                <div class="metadata-value">{self.data['metadata']['compute_capability']}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">CUDA Version</div>
                <div class="metadata-value">{self.data['metadata']['cuda_version']}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">CuPy Version</div>
                <div class="metadata-value">{self.data['metadata']['cupy_version']}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Timestamp</div>
                <div class="metadata-value">{self.data['metadata']['timestamp']}</div>
            </div>
        </div>
    </div>

    <div class="chart-container" id="chart1"></div>
    <div class="chart-container" id="chart2"></div>
    <div class="chart-container" id="chart3"></div>
    <div class="chart-container" id="chart4"></div>
    <div class="chart-container" id="chart5"></div>

    <script>
        var chart1 = {charts[0].to_json()};
        var chart2 = {charts[1].to_json()};
        var chart3 = {charts[2].to_json()};
        var chart4 = {charts[3].to_json()};
        var chart5 = {charts[4].to_json()};

        Plotly.newPlot('chart1', chart1.data, chart1.layout, {{responsive: true}});
        Plotly.newPlot('chart2', chart2.data, chart2.layout, {{responsive: true}});
        Plotly.newPlot('chart3', chart3.data, chart3.layout, {{responsive: true}});
        Plotly.newPlot('chart4', chart4.data, chart4.layout, {{responsive: true}});
        Plotly.newPlot('chart5', chart5.data, chart5.layout, {{responsive: true}});
    </script>
</body>
</html>
"""

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"Dashboard saved to: {output_path}")
        return output_path

    def save_data(self, output_file: str = "profiling_data.json"):
        """Save profiling data to JSON file."""
        output_path = self.output_dir / output_file

        with open(output_path, 'w') as f:
            json.dump(self.data, f, indent=2)

        print(f"Data saved to: {output_path}")
        return output_path

    def load_data(self, input_file: str):
        """Load profiling data from JSON file."""
        with open(input_file, 'r') as f:
            self.data = json.load(f)

        print(f"Data loaded from: {input_file}")
        return self.data


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Create performance visualization dashboard')
    parser.add_argument('--run-profiling', action='store_true',
                      help='Run profiling before creating visualizations')
    parser.add_argument('--data', type=str,
                      help='Load data from JSON file instead of running profiling')
    parser.add_argument('--output', type=str, default='performance_dashboard.html',
                      help='Output HTML file name')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size for profiling')
    parser.add_argument('--seq-len', type=int, default=64,
                      help='Sequence length for profiling')
    parser.add_argument('--output-dir', type=str, default='profiling_results',
                      help='Output directory for results')

    args = parser.parse_args()

    # Create dashboard
    dashboard = PerformanceDashboard(output_dir=args.output_dir)

    # Load or run profiling
    if args.data:
        dashboard.load_data(args.data)
    elif args.run_profiling:
        dashboard.run_profiling(batch_size=args.batch_size, seq_len=args.seq_len)
        dashboard.save_data()
    else:
        # Default: run profiling
        dashboard.run_profiling(batch_size=args.batch_size, seq_len=args.seq_len)
        dashboard.save_data()

    # Create visualizations
    dashboard_path = dashboard.create_dashboard(output_file=args.output)

    print("\n" + "=" * 80)
    print("✅ Dashboard Generation Complete!")
    print("=" * 80)
    print(f"\nOpen the dashboard in your browser:")
    print(f"  file://{dashboard_path.absolute()}")
    print("\nData files:")
    print(f"  JSON: {dashboard.output_dir / 'profiling_data.json'}")
    print(f"  HTML: {dashboard_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
