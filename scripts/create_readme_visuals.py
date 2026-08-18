#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the small, theme-aware SVGs embedded in the project README.

The benchmark visual reads the checked-in JSON instead of duplicating numbers,
so changing the benchmark data and rerunning this script keeps the README honest.
Only the Python standard library is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "docs/assets/comprehensive_comparison.json"
OUTPUT_DIR = ROOT / "docs/assets/readme"

FONT = (
    "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "
    "'Segoe UI', sans-serif"
)

THEMES = {
    "light": {
        "bg": "#F6F8FC",
        "panel": "#FFFFFF",
        "raised": "#EEF3F9",
        "border": "#D8E1ED",
        "text": "#142033",
        "muted": "#62728A",
        "primary": "#087EA4",
        "secondary": "#6D5BD0",
        "accent": "#168F5A",
        "warning": "#A5680A",
        "danger": "#B94A60",
        "shadow": "#26364D",
    },
    "dark": {
        "bg": "#0B1020",
        "panel": "#121A2D",
        "raised": "#19243A",
        "border": "#2A3855",
        "text": "#F3F7FC",
        "muted": "#A3B0C4",
        "primary": "#70D9EC",
        "secondary": "#A99AF4",
        "accent": "#64D6A0",
        "warning": "#F2B84B",
        "danger": "#F27D92",
        "shadow": "#000000",
    },
}


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 16,
    fill: str,
    weight: int = 400,
    anchor: str = "start",
    opacity: float = 1.0,
    letter_spacing: float = 0,
) -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{fill}" font-family="{FONT}" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'opacity="{opacity}" letter-spacing="{letter_spacing}">'
        f"{escape(value)}</text>"
    )


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str = "none",
    radius: float = 16,
    stroke_width: float = 1,
    opacity: float = 1.0,
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}" opacity="{opacity}"/>'
    )


def chip(x: float, y: float, width: float, label: str, theme: dict[str, str]) -> str:
    return "".join(
        [
            rect(x, y, width, 31, fill=theme["raised"], radius=8),
            text(x + 12, y + 21, label, size=13, fill=theme["muted"], weight=600),
        ]
    )


def arrow(x1: float, x2: float, y: float, theme: dict[str, str]) -> str:
    return (
        f'<path d="M {x1} {y} H {x2 - 9}" fill="none" '
        f'stroke="{theme["primary"]}" stroke-width="2" stroke-linecap="round"/>'
        f'<path d="M {x2 - 9} {y - 6} L {x2} {y} L {x2 - 9} {y + 6}" '
        f'fill="none" stroke="{theme["primary"]}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def svg_header(width: int, height: int, title_value: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{escape(title_value)}</title>",
        f"<desc id=\"desc\">{escape(description)}</desc>",
    ]


def architecture_svg(theme: dict[str, str]) -> str:
    width, height = 1200, 500
    out = svg_header(
        width,
        height,
        "cutileGPT tile-native inference pipeline",
        "A four-stage flow from Hugging Face checkpoint through TransformerLM and "
        "cuTile kernels to compiler-mapped tiles on an NVIDIA GPU.",
    )
    out += [
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">',
        f'<feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="{theme["shadow"]}" flood-opacity="0.12"/>',
        "</filter>",
        '<linearGradient id="tileGradient" x1="0" y1="0" x2="1" y2="1">',
        f'<stop offset="0" stop-color="{theme["primary"]}"/>',
        f'<stop offset="1" stop-color="{theme["secondary"]}"/>',
        "</linearGradient>",
        "</defs>",
        rect(0, 0, width, height, fill=theme["bg"], radius=26),
        text(54, 48, "TILE-NATIVE INFERENCE", size=12, fill=theme["primary"], weight=800, letter_spacing=1.8),
        text(54, 86, "From checkpoint to GPU, without leaving Python", size=28, fill=theme["text"], weight=750),
        text(54, 116, "You describe the tiles. The compiler maps the machine.", size=15, fill=theme["muted"]),
    ]

    panels = [
        (48, "01", "HF checkpoint", "Architecture + weights"),
        (330, "02", "TransformerLM", "Strict graph assembly"),
        (612, "03", "cuTile kernels", "Declarative tile math"),
        (894, "04", "NVIDIA GPU", "Compiler-owned mapping"),
    ]
    panel_y, panel_w, panel_h = 152, 258, 276
    for x, number, title_value, subtitle in panels:
        out += [
            f'<g filter="url(#shadow)">{rect(x, panel_y, panel_w, panel_h, fill=theme["panel"], stroke=theme["border"], radius=18)}</g>',
            rect(x + 18, panel_y + 18, 38, 26, fill=theme["raised"], radius=13),
            text(x + 37, panel_y + 36, number, size=11, fill=theme["primary"], weight=800, anchor="middle"),
            text(x + 18, panel_y + 78, title_value, size=21, fill=theme["text"], weight=750),
            text(x + 18, panel_y + 102, subtitle, size=13, fill=theme["muted"]),
        ]

    out += [
        arrow(306, 327, 289, theme),
        arrow(588, 609, 289, theme),
        arrow(870, 891, 289, theme),
    ]

    # Checkpoint panel.
    out += [
        chip(66, 278, 112, "config.json", theme),
        chip(184, 278, 104, "tokenizer", theme),
        chip(66, 317, 222, "*.safetensors · bf16", theme),
        chip(66, 356, 104, "Qwen", theme),
        chip(176, 356, 112, "Llama · Phi", theme),
    ]

    # Transformer panel.
    out += [
        chip(348, 278, 102, "RMSNorm", theme),
        chip(456, 278, 114, "RoPE", theme),
        chip(348, 317, 102, "GQA", theme),
        chip(456, 317, 114, "SwiGLU", theme),
        chip(348, 356, 222, "KV cache · fail closed", theme),
    ]

    # Kernel panel.
    out += [
        chip(630, 278, 204, "ct.load  →  compute", theme),
        chip(630, 317, 204, "online softmax", theme),
        chip(630, 356, 204, "ct.store  →  output", theme),
    ]

    # GPU tile grid.
    tile_x, tile_y, tile_size, gap = 920, 272, 27, 7
    for row in range(3):
        for col in range(6):
            opacity = 1.0 if (row + col) % 3 else 0.58
            out.append(
                rect(
                    tile_x + col * (tile_size + gap),
                    tile_y + row * (tile_size + gap),
                    tile_size,
                    tile_size,
                    fill="url(#tileGradient)",
                    radius=6,
                    opacity=opacity,
                )
            )
    out += [
        text(920, 392, "threads · memory · sync", size=12, fill=theme["muted"], weight=600),
        text(600, 468, "Readable model code  ·  specialized GPU execution", size=14, fill=theme["muted"], weight=650, anchor="middle"),
        "</svg>",
    ]
    return "\n".join(out) + "\n"


def heat_color(value: float, theme_name: str) -> str:
    palettes = {
        "light": ["#B94A60", "#A5680A", "#287184", "#187566", "#0B7D62"],
        "dark": ["#7D3E52", "#70561E", "#285D70", "#24695E", "#16866A"],
    }
    if value >= 1.0:
        index = 4
    elif value >= 0.95:
        index = 3
    elif value >= 0.85:
        index = 2
    elif value >= 0.70:
        index = 1
    else:
        index = 0
    return palettes[theme_name][index]


def benchmark_svg(theme_name: str, theme: dict[str, str], results: list[dict]) -> str:
    width, height = 1200, 660
    out = svg_header(
        width,
        height,
        "cutileGPT benchmark overview",
        "Three heatmaps show PyTorch latency divided by cutileGPT latency for "
        "nano, small, and medium models across four batches and three sequence lengths.",
    )
    out += [
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">',
        f'<feDropShadow dx="0" dy="7" stdDeviation="11" flood-color="{theme["shadow"]}" flood-opacity="0.12"/>',
        "</filter>",
        "</defs>",
        rect(0, 0, width, height, fill=theme["bg"], radius=26),
        text(54, 48, "36-CONFIGURATION BENCHMARK", size=12, fill=theme["primary"], weight=800, letter_spacing=1.8),
        text(54, 87, "End-to-end performance moves toward parity with more work", size=27, fill=theme["text"], weight=750),
        text(54, 116, "PyTorch latency ÷ cutileGPT latency · NVIDIA GB10 · higher is better", size=15, fill=theme["muted"]),
    ]

    by_key = {
        (item["model"], item["batch_size"], item["seq_len"]): item["speedup"]
        for item in results
    }
    model_info = [
        ("nano", "Nano", "3 layers · 48 hidden"),
        ("small", "Small", "6 layers · 384 hidden"),
        ("medium", "Medium", "8 layers · 512 hidden"),
    ]
    seqs = [64, 128, 256]
    batches = [1, 4, 8, 16]
    panel_xs = [48, 420, 792]
    panel_y, panel_w, panel_h = 154, 360, 390
    cell_w, cell_h, cell_gap = 80, 58, 8

    for (model_key, label, shape), panel_x in zip(model_info, panel_xs, strict=True):
        best = max(by_key[(model_key, batch, seq)] for batch in batches for seq in seqs)
        out += [
            f'<g filter="url(#shadow)">{rect(panel_x, panel_y, panel_w, panel_h, fill=theme["panel"], stroke=theme["border"], radius=18)}</g>',
            text(panel_x + 22, panel_y + 38, label, size=21, fill=theme["text"], weight=750),
            text(
                panel_x + panel_w - 22,
                panel_y + 36,
                f"best  {best:.3f}x",
                size=11,
                fill=theme["primary"],
                weight=800,
                anchor="end",
            ),
            text(panel_x + 22, panel_y + 61, shape, size=12, fill=theme["muted"]),
            text(panel_x + 58, panel_y + 95, "batch", size=11, fill=theme["muted"], weight=700, anchor="end"),
        ]

        grid_x = panel_x + 78
        grid_y = panel_y + 116
        for col, seq in enumerate(seqs):
            out.append(
                text(
                    grid_x + col * (cell_w + cell_gap) + cell_w / 2,
                    panel_y + 96,
                    f"seq {seq}",
                    size=11,
                    fill=theme["muted"],
                    weight=700,
                    anchor="middle",
                )
            )
        for row, batch in enumerate(batches):
            y = grid_y + row * (cell_h + cell_gap)
            out.append(
                text(panel_x + 58, y + 35, str(batch), size=13, fill=theme["muted"], weight=700, anchor="end")
            )
            for col, seq in enumerate(seqs):
                value = by_key[(model_key, batch, seq)]
                x = grid_x + col * (cell_w + cell_gap)
                stroke = theme["accent"] if value >= 1.0 else "none"
                stroke_width = 3 if value >= 1.0 else 1
                out += [
                    rect(
                        x,
                        y,
                        cell_w,
                        cell_h,
                        fill=heat_color(value, theme_name),
                        stroke=stroke,
                        radius=10,
                        stroke_width=stroke_width,
                    ),
                    text(
                        x + cell_w / 2,
                        y + 35,
                        f"{value:.3f}x",
                        size=14,
                        fill="#FFFFFF",
                        weight=800,
                        anchor="middle",
                    ),
                ]

    legend_y = 586
    legend = [
        ("< 0.70", 0.60),
        ("0.70–0.84", 0.75),
        ("0.85–0.94", 0.90),
        ("0.95–0.99", 0.97),
        ("≥ 1.00", 1.01),
    ]
    start_x = 208
    for index, (label, value) in enumerate(legend):
        x = start_x + index * 160
        out += [
            rect(x, legend_y, 24, 16, fill=heat_color(value, theme_name), radius=5),
            text(x + 34, legend_y + 13, label, size=11, fill=theme["muted"], weight=650),
        ]
    out += [
        text(600, 635, "Ratio ≥ 1.0 means cutileGPT is faster for that configuration", size=12, fill=theme["muted"], anchor="middle"),
        "</svg>",
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    results = json.loads(DATA_PATH.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for theme_name, theme in THEMES.items():
        (OUTPUT_DIR / f"tile-pipeline-{theme_name}.svg").write_text(
            architecture_svg(theme)
        )
        (OUTPUT_DIR / f"benchmark-overview-{theme_name}.svg").write_text(
            benchmark_svg(theme_name, theme, results)
        )

    print(f"Wrote 4 README visuals to {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
