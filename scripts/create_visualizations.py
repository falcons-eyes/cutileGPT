#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Create performance visualizations for cutileGPT README.

This script creates publication-quality visualizations using:
- Actual benchmark data from our GPU
- Plotly for interactive charts
- Kaleido for PNG/SVG export
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path


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

        fig = go.Figure()

        # Add bars
        fig.add_trace(go.Bar(
            name='Execution Time',
            x=['CuPy Kernel', 'Tile Kernel'],
            y=[data['cupy_kernel_ms'], data['tile_kernel_ms']],
            marker_color=['#ef4444', '#10b981'],
            text=[f"{data['cupy_kernel_ms']:.3f} ms", f"{data['tile_kernel_ms']:.3f} ms"],
            textposition='outside',
            textfont=dict(size=14, color='black', family='Arial Black')
        ))

        fig.update_layout(
            title=dict(
                text=f'<b>GELU Kernel: {data["speedup"]:.1f}x Speedup</b>',
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis=dict(
                title=dict(text='<b>Implementation</b>', font=dict(size=16, family='Arial Black')),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title=dict(text='<b>Latency (ms)</b>', font=dict(size=16, family='Arial Black')),
                tickfont=dict(size=14)
            ),
            template='plotly_white',
            height=500,
            width=800,
            showlegend=False,
            margin=dict(t=80, b=80, l=80, r=80),
            annotations=[
                dict(
                    text=f'Tensor: {data["tensor_shape"]}<br>{data["total_elements"]:,} elements',
                    xref='paper', yref='paper',
                    x=0.5, y=1.15,
                    showarrow=False,
                    font=dict(size=12, color='gray')
                )
            ]
        )

        return fig

    def create_cutile_performance_chart(self):
        """Create cutileGPT internal performance chart."""
        data = self.data['cutile_benchmarks']

        models = list(data.keys())
        latencies = [data[m]['latency_ms'] for m in models]
        throughputs = [data[m]['throughput'] / 1000 for m in models]  # Convert to K tok/s

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('<b>Latency</b>', '<b>Throughput</b>'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )

        # Latency chart
        fig.add_trace(
            go.Bar(
                x=models,
                y=latencies,
                marker_color='#3b82f6',
                text=[f"{l:.2f} ms" for l in latencies],
                textposition='outside',
                textfont=dict(size=12, color='black', family='Arial'),
                name='Latency'
            ),
            row=1, col=1
        )

        # Throughput chart
        fig.add_trace(
            go.Bar(
                x=models,
                y=throughputs,
                marker_color='#10b981',
                text=[f"{t:.0f}K" for t in throughputs],
                textposition='outside',
                textfont=dict(size=12, color='black', family='Arial'),
                name='Throughput'
            ),
            row=1, col=2
        )

        fig.update_xaxes(title=dict(text="<b>Model Config</b>", font=dict(size=14, family='Arial Black')), row=1, col=1)
        fig.update_xaxes(title=dict(text="<b>Model Config</b>", font=dict(size=14, family='Arial Black')), row=1, col=2)
        fig.update_yaxes(title=dict(text="<b>ms</b>", font=dict(size=14, family='Arial Black')), row=1, col=1)
        fig.update_yaxes(title=dict(text="<b>K tokens/sec</b>", font=dict(size=14, family='Arial Black')), row=1, col=2)

        fig.update_layout(
            title=dict(
                text='<b>cutileGPT Performance (Batch=4, Seq=64)</b>',
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            template='plotly_white',
            height=500,
            width=1000,
            showlegend=False,
            margin=dict(t=100, b=80, l=80, r=80)
        )

        return fig

    def create_pytorch_comparison_chart(self):
        """Create PyTorch vs cutileGPT comparison chart."""
        data = self.data['pytorch_comparison']

        models = ['Nano (3 layers)', 'Medium (6 layers)']
        pytorch_times = [data['nano']['pytorch_ms'], data['medium']['pytorch_ms']]
        cutile_times = [data['nano']['cutile_ms'], data['medium']['cutile_ms']]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='PyTorch minGPT',
            x=models,
            y=pytorch_times,
            marker_color='#f97316',
            text=[f"{t:.2f} ms" for t in pytorch_times],
            textposition='outside',
            textfont=dict(size=12, color='black', family='Arial')
        ))

        fig.add_trace(go.Bar(
            name='cutileGPT',
            x=models,
            y=cutile_times,
            marker_color='#3b82f6',
            text=[f"{t:.2f} ms" for t in cutile_times],
            textposition='outside',
            textfont=dict(size=12, color='black', family='Arial')
        ))

        fig.update_layout(
            title=dict(
                text='<b>PyTorch minGPT vs cutileGPT</b>',
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis=dict(
                title=dict(text='<b>Model Configuration</b>', font=dict(size=16, family='Arial Black')),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title=dict(text='<b>Latency (ms)</b>', font=dict(size=16, family='Arial Black')),
                tickfont=dict(size=14)
            ),
            barmode='group',
            template='plotly_white',
            height=500,
            width=900,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                font=dict(size=14, family='Arial')
            ),
            margin=dict(t=100, b=80, l=80, r=80),
            annotations=[
                dict(
                    text='Batch=8, Seq=128<br>Lower is better',
                    xref='paper', yref='paper',
                    x=0.5, y=1.15,
                    showarrow=False,
                    font=dict(size=12, color='gray')
                )
            ]
        )

        return fig

    def create_tile_philosophy_diagram(self):
        """Create Tile Programming Philosophy comparison diagram."""
        fig = go.Figure()

        # Add comparison table as annotations
        categories = ['Thread Management', 'Synchronization', 'Code Complexity', 'Portability']
        traditional = ['Manual (threadIdx.x)', 'Explicit __syncthreads()', '~150 lines/kernel', 'GPU-specific']
        tile = ['Compiler handles', 'Automatic dependencies', '~20 lines/kernel', 'Hardware portable']

        # Create a clean comparison layout
        fig.add_trace(go.Table(
            header=dict(
                values=['<b>Aspect</b>', '<b>Traditional CUDA</b>', '<b>Tile Programming</b>'],
                fill_color='#667eea',
                align='left',
                font=dict(color='white', size=14, family='Arial Black'),
                height=40
            ),
            cells=dict(
                values=[categories, traditional, tile],
                fill_color=[['white']*4, ['#ffebee']*4, ['#e8f5e9']*4],
                align='left',
                font=dict(size=12, family='Arial'),
                height=35
            )
        ))

        fig.update_layout(
            title=dict(
                text='<b>Traditional CUDA vs Tile Programming</b>',
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            height=400,
            width=1000,
            margin=dict(t=80, b=40, l=40, r=40)
        )

        return fig

    def create_all_visualizations(self):
        """Create all visualizations and export as PNG."""
        print("Creating visualizations...")

        charts = {
            'gelu_speedup': self.create_gelu_speedup_chart(),
            'cutile_performance': self.create_cutile_performance_chart(),
            'pytorch_comparison': self.create_pytorch_comparison_chart(),
            'tile_philosophy': self.create_tile_philosophy_diagram()
        }

        for name, fig in charts.items():
            # Export as PNG
            png_path = self.output_dir / f"{name}.png"
            fig.write_image(str(png_path), format='png', scale=2)
            print(f"  ✅ {png_path}")

            # Also save as HTML for interactive viewing
            html_path = self.output_dir / f"{name}.html"
            fig.write_html(str(html_path))
            print(f"  📊 {html_path}")

        # Save data as JSON
        data_path = self.output_dir / "benchmark_data.json"
        with open(data_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"\n  💾 Data saved to {data_path}")

        print("\n✅ All visualizations created!")
        return charts


def main():
    """Main entry point."""
    creator = VisualizationCreator()
    creator.create_all_visualizations()


if __name__ == "__main__":
    main()
