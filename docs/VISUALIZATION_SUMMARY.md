# Visualization System Implementation Summary

## What Was Implemented

Modern interactive visualization system for cutileGPT performance analysis, replacing manual SVG/MD generation with **Plotly** - a popular, modern visualization library used in 2026.

---

## New Files Created

### 1. `visualize_performance.py` (18KB)
**Purpose**: Generate interactive performance dashboards for cutileGPT profiling data

**Features**:
- Automated profiling across all model configs (nano, micro, mini, tile-small, tile-medium, tile-large)
- GPU-accurate timing using CuPy events
- 5 interactive visualizations:
  - Forward pass latency comparison
  - Throughput analysis (tokens/sec)
  - Latency percentile distribution (min, median, p95, p99, max)
  - Model architecture comparison
  - Efficiency metrics (throughput per layer)
- JSON data export for analysis
- Standalone HTML dashboard (no server required)

**Usage**:
```bash
# Run profiling and generate dashboard
uv run python visualize_performance.py --run-profiling

# Generate from existing data
uv run python visualize_performance.py --data profiling_results/profiling_data.json

# Custom configuration
uv run python visualize_performance.py --run-profiling --batch-size 8 --seq-len 128
```

### 2. `visualize_comparison.py` (16KB)
**Purpose**: AS-IS (minGPT) vs TO-BE (cutileGPT) comparison visualization

**Features**:
- Side-by-side performance comparison
- Speedup calculation and visualization
- Interactive charts with error bars
- Comparison summary table
- Support for single or all model configs

**Usage**:
```bash
# Compare single model
uv run python visualize_comparison.py --model tile-medium --benchmark

# Compare all models
uv run python visualize_comparison.py --compare-all --benchmark
```

### 3. `VISUALIZATION_GUIDE.md`
**Purpose**: Complete user guide for the visualization system

**Contents**:
- Quick start guide
- Output file descriptions
- Visualization explanations
- Data analysis examples
- Command reference
- Integration with existing tools
- Troubleshooting guide

### 4. `pyproject.toml` (Updated)
**Added dependencies**:
```toml
"plotly>=5.24.0",      # Modern interactive visualizations
"kaleido>=0.2.1",      # Static image export (PNG, PDF)
```

---

## Generated Output Files

### Performance Dashboard
**Location**: `profiling_results/performance_dashboard.html` (43KB)

**System Information Panel**:
- GPU: NVIDIA GB10
- Compute Capability: 12.1
- CUDA Version: 13000
- CuPy Version: 13.6.0
- Timestamp: 2026-01-26

**5 Interactive Charts**:
1. **Latency Comparison**: Bar chart with error bars
2. **Throughput**: Tokens/sec across configs
3. **Percentile Distribution**: Multi-line chart (min → p99)
4. **Model Architecture**: 3-panel comparison (layers, dims, heads)
5. **Efficiency**: Scatter plot (throughput per layer)

### Profiling Data
**Location**: `profiling_results/profiling_data.json` (3.3KB)

**Structure**:
```json
{
  "metadata": {
    "timestamp": "2026-01-26T14:27:35.233860",
    "gpu_name": "NVIDIA GB10",
    "compute_capability": "12.1",
    ...
  },
  "benchmarks": {
    "gpt_nano": {
      "config": {"n_layer": 3, "n_head": 3, "n_embd": 48},
      "timing": {
        "mean": 1.32, "std": 0.11, "min": 1.24, "max": 1.75,
        "median": 1.28, "p95": 1.63, "p99": 1.74
      },
      "throughput_tokens_per_sec": 193731
    },
    ...
  }
}
```

---

## Performance Results

### Benchmark Summary (Batch=4, Seq=64)

| Model | Latency (ms) | Throughput (tok/s) | Config |
|-------|-------------|-------------------|---------|
| **gpt_nano** | 1.32 ± 0.11 | 193.7K | 3L, 3H, 48D |
| **gpt_micro** | 3.70 ± 0.42 | 69.2K | 4L, 4H, 128D |
| **gpt_mini** | 11.50 ± 0.72 | 22.3K | 6L, 6H, 192D |
| **gpt_tile_small** | 1.50 ± 0.09 | 171.1K | 4L, 4H, 64D |
| **gpt_tile_medium** | 5.27 ± 0.69 | 48.6K | 6L, 4H, 128D |
| **gpt_tile_large** | 29.31 ± 0.80 | 8.7K | 8L, 8H, 256D |

**Fastest Config**: gpt_nano (1.32ms, 193.7K tok/s)
**Best Tile Config**: gpt_tile_small (1.50ms, 171.1K tok/s)

---

## Why Plotly?

### Advantages Over Manual SVG/MD

| Aspect | Manual SVG/MD | Plotly Dashboard |
|--------|--------------|------------------|
| **Interactivity** | ❌ Static images | ✅ Hover, zoom, pan, filter |
| **Data Updates** | ❌ Manual XML editing | ✅ Automatic from JSON |
| **Customization** | ❌ Complex SVG paths | ✅ Python API, one-liners |
| **Export Formats** | SVG only | HTML, PNG, PDF, SVG |
| **Data Persistence** | Embedded in SVG | Separate structured JSON |
| **Maintenance** | High (manual edits) | Low (data-driven) |
| **Modern UX** | Basic CSS | Responsive, professional |
| **Analytics** | Requires manual calc | Built-in statistics |

### Popularity (2026)

Plotly is widely used for:
- Data science dashboards
- Performance monitoring
- ML experiment tracking
- Interactive reports
- Web-based analytics

**Alternative libraries considered**:
- Altair (declarative, but less interactive)
- Bokeh (older, declining usage)
- Streamlit (requires server)
- Matplotlib (not modern/interactive)

---

## Key Features

### 1. Data-Driven Architecture
```
profile_performance.py → JSON data → visualize_performance.py → HTML dashboard
                           ↓
                    Reusable, analyzable
```

### 2. Interactive Exploration
- **Hover**: See exact values
- **Zoom**: Focus on specific ranges
- **Pan**: Navigate large datasets
- **Toggle**: Show/hide series
- **Export**: Download as PNG

### 3. Code-Based Layout
All visualizations defined in Python:
```python
fig = go.Figure()
fig.add_trace(go.Bar(x=configs, y=latencies))
fig.update_layout(title='Performance', template='plotly_white')
```

### 4. Standalone Output
HTML files include:
- Plotly.js library (CDN)
- All chart data (embedded JSON)
- Custom CSS styling
- No backend required

---

## Integration Points

### With Existing Scripts

```
profile_performance.py          → Terminal output
visualize_performance.py        → JSON + HTML
test_text_generation.py         → Text generation
scripts/run_nsys_profile.sh     → nsys profiling
scripts/run_ncu_profile.sh      → ncu profiling
cutile_gpt/compare.py           → AS-IS vs TO-BE
visualize_comparison.py         → AS-IS vs TO-BE viz
```

### Workflow Example

```bash
# 1. Run comprehensive profiling
uv run python visualize_performance.py --run-profiling

# 2. Open dashboard
open profiling_results/performance_dashboard.html

# 3. Run AS-IS vs TO-BE comparison
uv run python visualize_comparison.py --compare-all --benchmark

# 4. Open comparison dashboard
open profiling_results/comparison_dashboard.html

# 5. Run nsys for detailed kernel analysis
bash scripts/run_nsys_profile.sh

# 6. Analyze nsys results
nsys-ui profiling_results/cutile_nsys.nsys-rep
```

---

## Data Analysis Examples

### Load and Analyze

```python
import json

# Load data
with open('profiling_results/profiling_data.json', 'r') as f:
    data = json.load(f)

# Find fastest config
configs = data['benchmarks']
fastest = min(configs.items(), key=lambda x: x[1]['timing']['mean'])
print(f"Fastest: {fastest[0]} at {fastest[1]['timing']['mean']:.2f}ms")

# Compare tile vs non-tile
tile_avg = sum(v['timing']['mean'] for k, v in configs.items() if 'tile' in k) / 3
standard_avg = sum(v['timing']['mean'] for k, v in configs.items() if 'tile' not in k) / 3
print(f"Tile avg: {tile_avg:.2f}ms, Standard avg: {standard_avg:.2f}ms")
```

### Custom Visualization

```python
from visualize_performance import PerformanceDashboard
import plotly.graph_objects as go

# Load existing data
dashboard = PerformanceDashboard()
dashboard.load_data('profiling_results/profiling_data.json')

# Create custom chart
configs = list(dashboard.data['benchmarks'].keys())
p99s = [dashboard.data['benchmarks'][c]['timing']['p99'] for c in configs]

fig = go.Figure(go.Bar(x=configs, y=p99s))
fig.update_layout(title='P99 Latency Analysis')
fig.show()
```

---

## Next Steps

### Recommended Enhancements

1. **Memory Profiling**:
   - Add GPU memory usage tracking
   - Visualize memory bandwidth utilization
   - Compare memory efficiency across configs

2. **Kernel Analysis Integration**:
   - Parse nsys/ncu output
   - Visualize kernel timelines
   - Identify bottlenecks

3. **Historical Tracking**:
   - Store results over time
   - Track performance regressions
   - Visualize trends

4. **Comparison Mode**:
   - Compare different GPUs
   - Compare CUDA versions
   - Compare optimization flags

5. **Export to Reports**:
   - Generate PDF reports
   - Export to PowerPoint
   - Create static images for docs

---

## Quick Reference

### Commands

```bash
# Performance profiling
uv run python visualize_performance.py --run-profiling

# AS-IS vs TO-BE comparison
uv run python visualize_comparison.py --compare-all --benchmark

# Custom batch/sequence
uv run python visualize_performance.py --run-profiling --batch-size 16 --seq-len 256

# Generate from existing data
uv run python visualize_performance.py --data profiling_results/profiling_data.json
```

### Files

```
visualize_performance.py       - Performance dashboard generator
visualize_comparison.py        - AS-IS vs TO-BE comparison
VISUALIZATION_GUIDE.md        - Complete user guide
profiling_results/
  ├─ performance_dashboard.html   - Interactive dashboard
  ├─ profiling_data.json          - Structured data
  ├─ comparison_dashboard.html    - AS-IS vs TO-BE viz
  └─ comparison_data.json         - Comparison data
```

---

## Summary

✅ **Implemented**: Modern visualization system with Plotly
✅ **Created**: 2 visualization scripts, comprehensive guide
✅ **Generated**: Interactive HTML dashboards with 8 charts
✅ **Stored**: Structured JSON data for analysis
✅ **Benefits**: Interactive, maintainable, data-driven, modern UX

The visualization system is production-ready and significantly improves upon manual SVG/MD generation.
