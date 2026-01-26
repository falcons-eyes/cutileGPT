# cutileGPT Visualization Guide

Modern interactive performance visualization for cutileGPT using Plotly.

## Overview

The visualization system provides interactive dashboards to analyze cutileGPT performance metrics across different model configurations. It uses **Plotly**, a modern visualization library that creates interactive, web-based charts.

## Features

- **Interactive Charts**: Hover, zoom, pan, and explore data dynamically
- **Multiple Visualizations**:
  - Forward pass latency comparison
  - Throughput analysis (tokens/sec)
  - Percentile distribution (min, median, p95, p99, max)
  - Model architecture comparison
  - Efficiency metrics (throughput per layer)
- **Data Persistence**: JSON format for easy storage and analysis
- **Standalone HTML**: No server required, open directly in browser

## Quick Start

### 1. Run Profiling and Generate Dashboard

```bash
# Run full profiling and create visualization
uv run python visualize_performance.py --run-profiling

# Open the generated dashboard
open profiling_results/performance_dashboard.html
```

### 2. Generate Dashboard from Existing Data

```bash
# If you already have profiling_data.json
uv run python visualize_performance.py --data profiling_results/profiling_data.json
```

### 3. Custom Configuration

```bash
# Custom batch size and sequence length
uv run python visualize_performance.py --run-profiling --batch-size 8 --seq-len 128

# Custom output location
uv run python visualize_performance.py --run-profiling --output my_dashboard.html --output-dir my_results
```

## Output Files

### `profiling_data.json`
Complete profiling results in JSON format:

```json
{
  "metadata": {
    "timestamp": "2026-01-26T14:27:35.233860",
    "cupy_version": "13.6.0",
    "cuda_version": "13000",
    "gpu_name": "NVIDIA GB10",
    "compute_capability": "12.1"
  },
  "benchmarks": {
    "gpt_nano": {
      "config": {...},
      "timing": {
        "mean": 1.32,
        "std": 0.11,
        "min": 1.24,
        "max": 1.75,
        "median": 1.28,
        "p95": 1.63,
        "p99": 1.74
      },
      "throughput_tokens_per_sec": 193731.2
    },
    ...
  }
}
```

### `performance_dashboard.html`
Interactive HTML dashboard with:
- System information panel
- 5 interactive charts
- Responsive design
- No external dependencies (standalone file)

## Visualizations Explained

### 1. Forward Pass Latency Comparison
Bar chart showing mean latency across model configs with error bars (standard deviation) and min/max markers.

**Use case**: Compare absolute performance across different model sizes.

### 2. Throughput Comparison
Bar chart showing tokens processed per second.

**Use case**: Identify which configs deliver the highest throughput.

### 3. Latency Percentile Distribution
Multi-line chart showing latency distribution (min, median, p95, p99, max).

**Use case**: Understand latency variability and tail latencies.

### 4. Model Architecture Comparison
Three-panel bar chart showing layers, embedding dimensions, and attention heads.

**Use case**: Understand model complexity and size trade-offs.

### 5. Efficiency Chart
Scatter plot showing throughput per layer vs. number of layers.

**Use case**: Evaluate efficiency relative to model complexity.

## Data Analysis

### Load and Analyze Data in Python

```python
import json

# Load profiling data
with open('profiling_results/profiling_data.json', 'r') as f:
    data = json.load(f)

# Access specific metrics
gpt_nano_latency = data['benchmarks']['gpt_nano']['timing']['mean']
gpt_nano_throughput = data['benchmarks']['gpt_nano']['throughput_tokens_per_sec']

print(f"Latency: {gpt_nano_latency:.2f} ms")
print(f"Throughput: {gpt_nano_throughput:.0f} tok/s")
```

### Compare Configurations

```python
# Compare tile-optimized vs standard configs
tile_configs = {k: v for k, v in data['benchmarks'].items() if 'tile' in k}
standard_configs = {k: v for k, v in data['benchmarks'].items() if 'tile' not in k}

print("Tile-optimized configs:")
for name, metrics in tile_configs.items():
    print(f"  {name}: {metrics['timing']['mean']:.2f} ms")
```

## Command Reference

### visualize_performance.py

```
Options:
  --run-profiling        Run profiling before creating visualizations
  --data FILE           Load data from JSON file instead of running profiling
  --output FILE         Output HTML file name (default: performance_dashboard.html)
  --batch-size N        Batch size for profiling (default: 4)
  --seq-len N          Sequence length for profiling (default: 64)
  --output-dir DIR     Output directory for results (default: profiling_results)

Examples:
  # Full profiling run
  uv run python visualize_performance.py --run-profiling

  # Large batch profiling
  uv run python visualize_performance.py --run-profiling --batch-size 16 --seq-len 256

  # Generate from existing data
  uv run python visualize_performance.py --data profiling_results/profiling_data.json
```

## Integration with Existing Tools

### With profile_performance.py

```bash
# Run detailed profiling
uv run python profile_performance.py

# Then visualize the results (manually load data)
# Note: profile_performance.py prints to stdout, not JSON
# Use visualize_performance.py --run-profiling for JSON output
```

### With nsys/ncu Profiling

```bash
# Run nsys profiling
bash scripts/run_nsys_profile.sh

# Run custom profiling and visualization
uv run python visualize_performance.py --run-profiling

# Both results are in profiling_results/
ls profiling_results/
# - cutile_nsys.nsys-rep (nsys data)
# - performance_dashboard.html (interactive viz)
# - profiling_data.json (structured data)
```

## Benefits Over Manual SVG/MD

| Feature | Manual SVG/MD | Plotly Dashboard |
|---------|--------------|------------------|
| Interactivity | ❌ Static | ✅ Hover, zoom, pan |
| Data Updates | ❌ Manual editing | ✅ Automatic from JSON |
| Customization | ❌ Complex XML | ✅ Simple Python API |
| Export Formats | SVG only | HTML, PNG, PDF, SVG |
| Data Storage | Embedded in SVG | Separate JSON file |
| Maintenance | High effort | Low effort |
| Styling | CSS in XML | Modern themes |

## Advanced Usage

### Custom Chart Creation

Extend `PerformanceDashboard` class to add custom visualizations:

```python
from visualize_performance import PerformanceDashboard
import plotly.graph_objects as go

class CustomDashboard(PerformanceDashboard):
    def create_custom_chart(self):
        """Add your custom visualization."""
        configs = list(self.data['benchmarks'].keys())
        # Your custom logic here

        fig = go.Figure(...)
        return fig

    def create_dashboard(self, output_file="custom_dashboard.html"):
        # Add your custom chart to the charts list
        charts = [
            self.create_latency_chart(),
            self.create_custom_chart(),  # Your chart
            # ... other charts
        ]
        # ... rest of dashboard creation
```

### Export Static Images

```python
# Requires kaleido (already installed)
from visualize_performance import PerformanceDashboard

dashboard = PerformanceDashboard()
dashboard.load_data('profiling_results/profiling_data.json')

# Create chart
fig = dashboard.create_latency_chart()

# Export to PNG
fig.write_image("latency_chart.png", width=1200, height=600)

# Export to PDF
fig.write_image("latency_chart.pdf")
```

## Troubleshooting

### Dashboard won't open
- Check file path: `file:///absolute/path/to/performance_dashboard.html`
- Try different browser (Chrome, Firefox, Safari)
- Check browser console for JavaScript errors

### Missing data
- Ensure profiling completed successfully
- Check `profiling_data.json` exists and is valid JSON
- Run with `--run-profiling` to regenerate

### Plotly import errors
```bash
# Reinstall dependencies
uv pip install --force-reinstall plotly kaleido
```

## Performance Tips

- **Large Batches**: Use `--batch-size 16` for throughput-focused benchmarks
- **Long Sequences**: Use `--seq-len 512` for sequence modeling workloads
- **More Iterations**: Edit `benchmark_forward_pass()` to increase iteration count for more stable results

## Next Steps

- Add comparison with baseline (minGPT) to dashboard
- Integrate nsys timeline data into visualizations
- Create A/B comparison mode for different GPU architectures
- Add memory usage tracking and visualization
