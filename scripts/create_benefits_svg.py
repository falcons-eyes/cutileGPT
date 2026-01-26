#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Create dramatic visual comparison SVG for Tile Programming benefits.

Shows the dramatic simplification: 150 lines → 20 lines (87% reduction)
"""

from pathlib import Path


def create_code_comparison_svg():
    """Create side-by-side code comparison showing dramatic simplification."""

    svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="1200" height="600" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="redGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#fee2e2;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#fecaca;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="greenGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#d1fae5;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#a7f3d0;stop-opacity:1" />
    </linearGradient>
    <filter id="shadow">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.3"/>
    </filter>
  </defs>

  <!-- Background -->
  <rect width="1200" height="600" fill="#f9fafb"/>

  <!-- Title -->
  <text x="600" y="40" font-family="Arial, sans-serif" font-size="32" font-weight="bold"
        text-anchor="middle" fill="#1f2937">
    Traditional CUDA vs Tile Programming
  </text>

  <!-- Traditional CUDA Side (LEFT) -->
  <g id="traditional">
    <!-- Box -->
    <rect x="50" y="80" width="500" height="480" rx="10"
          fill="url(#redGradient)" stroke="#ef4444" stroke-width="3" filter="url(#shadow)"/>

    <!-- Header -->
    <rect x="50" y="80" width="500" height="60" rx="10" fill="#ef4444"/>
    <text x="300" y="118" font-family="Arial, sans-serif" font-size="24" font-weight="bold"
          text-anchor="middle" fill="white">
      ❌ Traditional CUDA
    </text>

    <!-- Content -->
    <text x="80" y="170" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="#991b1b">
      ~150 lines of code
    </text>

    <!-- Code lines visualization (many lines) -->
    <g id="code-lines-traditional">
      <line x1="80" y1="190" x2="520" y2="190" stroke="#dc2626" stroke-width="2"/>
      <line x1="80" y1="205" x2="480" y2="205" stroke="#dc2626" stroke-width="2"/>
      <line x1="80" y1="220" x2="510" y2="220" stroke="#dc2626" stroke-width="2"/>
      <line x1="80" y1="235" x2="450" y2="235" stroke="#dc2626" stroke-width="2"/>
      <line x1="80" y1="250" x2="500" y2="250" stroke="#dc2626" stroke-width="2"/>
      <line x1="100" y1="265" x2="490" y2="265" stroke="#dc2626" stroke-width="2"/>
      <line x1="100" y1="280" x2="470" y2="280" stroke="#dc2626" stroke-width="2"/>
      <line x1="100" y1="295" x2="520" y2="295" stroke="#dc2626" stroke-width="2"/>
      <line x1="80" y1="310" x2="460" y2="310" stroke="#dc2626" stroke-width="2"/>
      <line x1="80" y1="325" x2="505" y2="325" stroke="#dc2626" stroke-width="2"/>
      <line x1="100" y1="340" x2="480" y2="340" stroke="#dc2626" stroke-width="2"/>
      <line x1="100" y1="355" x2="495" y2="355" stroke="#dc2626" stroke-width="2"/>
      <line x1="120" y1="370" x2="510" y2="370" stroke="#dc2626" stroke-width="2"/>
      <line x1="120" y1="385" x2="470" y2="385" stroke="#dc2626" stroke-width="2"/>
      <line x1="100" y1="400" x2="490" y2="400" stroke="#dc2626" stroke-width="2"/>
      <line x1="80" y1="415" x2="500" y2="415" stroke="#dc2626" stroke-width="2"/>
      <line x1="80" y1="430" x2="460" y2="430" stroke="#dc2626" stroke-width="2"/>
      <line x1="80" y1="445" x2="485" y2="445" stroke="#dc2626" stroke-width="2"/>
      <text x="300" y="475" font-family="monospace" font-size="14" text-anchor="middle" fill="#7f1d1d">
        ... and many more lines ...
      </text>
    </g>

    <!-- Complexity indicators -->
    <text x="80" y="510" font-family="Arial, sans-serif" font-size="14" fill="#7f1d1d">
      • Manual thread management
    </text>
    <text x="80" y="530" font-family="Arial, sans-serif" font-size="14" fill="#7f1d1d">
      • Explicit __syncthreads()
    </text>
    <text x="80" y="550" font-family="Arial, sans-serif" font-size="14" fill="#7f1d1d">
      • GPU-specific code
    </text>
  </g>

  <!-- Arrow and Reduction Label -->
  <g id="arrow">
    <path d="M 560 320 L 630 320 L 630 300 L 670 330 L 630 360 L 630 340 L 560 340 Z"
          fill="#10b981" stroke="#059669" stroke-width="2"/>
    <text x="615" y="285" font-family="Arial, sans-serif" font-size="20" font-weight="bold"
          text-anchor="middle" fill="#059669">
      87% Less
    </text>
    <text x="615" y="385" font-family="Arial, sans-serif" font-size="16" font-weight="bold"
          text-anchor="middle" fill="#059669">
      Code
    </text>
  </g>

  <!-- Tile Programming Side (RIGHT) -->
  <g id="tile">
    <!-- Box -->
    <rect x="680" y="80" width="470" height="480" rx="10"
          fill="url(#greenGradient)" stroke="#10b981" stroke-width="3" filter="url(#shadow)"/>

    <!-- Header -->
    <rect x="680" y="80" width="470" height="60" rx="10" fill="#10b981"/>
    <text x="915" y="118" font-family="Arial, sans-serif" font-size="24" font-weight="bold"
          text-anchor="middle" fill="white">
      ✅ Tile Programming
    </text>

    <!-- Content -->
    <text x="710" y="170" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="#065f46">
      ~20 lines of code
    </text>

    <!-- Code lines visualization (few lines) -->
    <g id="code-lines-tile">
      <line x1="710" y1="190" x2="1120" y2="190" stroke="#059669" stroke-width="2"/>
      <line x1="710" y1="210" x2="1080" y2="210" stroke="#059669" stroke-width="2"/>
      <line x1="710" y1="230" x2="1110" y2="230" stroke="#059669" stroke-width="2"/>
      <line x1="730" y1="250" x2="1100" y2="250" stroke="#059669" stroke-width="2"/>
      <line x1="730" y1="270" x2="1090" y2="270" stroke="#059669" stroke-width="2"/>
    </g>

    <text x="915" y="320" font-family="Arial, sans-serif" font-size="16" font-style="italic"
          text-anchor="middle" fill="#047857">
      Clean, declarative code
    </text>

    <!-- Benefits -->
    <g id="benefits">
      <rect x="710" y="360" width="410" height="180" rx="5" fill="white" opacity="0.7"/>

      <text x="730" y="390" font-family="Arial, sans-serif" font-size="15" font-weight="bold" fill="#065f46">
        ✓ Compiler handles threads
      </text>
      <text x="750" y="410" font-family="Arial, sans-serif" font-size="13" fill="#047857">
        No manual threadIdx.x
      </text>

      <text x="730" y="440" font-family="Arial, sans-serif" font-size="15" font-weight="bold" fill="#065f46">
        ✓ Automatic synchronization
      </text>
      <text x="750" y="460" font-family="Arial, sans-serif" font-size="13" fill="#047857">
        No __syncthreads()
      </text>

      <text x="730" y="490" font-family="Arial, sans-serif" font-size="15" font-weight="bold" fill="#065f46">
        ✓ Hardware portable
      </text>
      <text x="750" y="510" font-family="Arial, sans-serif" font-size="13" fill="#047857">
        Same code, any GPU
      </text>
    </g>
  </g>

  <!-- Bottom stats -->
  <rect x="400" y="570" width="400" height="20" fill="#6366f1" opacity="0.1" rx="10"/>
  <text x="600" y="585" font-family="Arial, sans-serif" font-size="14" font-weight="bold"
        text-anchor="middle" fill="#4f46e5">
    150 lines → 20 lines | 87% code reduction | Same performance
  </text>
</svg>'''

    return svg


def create_architecture_simplification_svg():
    """Create visual showing architecture simplification."""

    svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="1000" height="500" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="shadow">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.3"/>
    </filter>
  </defs>

  <!-- Background -->
  <rect width="1000" height="500" fill="#f9fafb"/>

  <!-- Title -->
  <text x="500" y="40" font-family="Arial, sans-serif" font-size="28" font-weight="bold"
        text-anchor="middle" fill="#1f2937">
    Architecture Complexity: Traditional vs Tile
  </text>

  <!-- Traditional CUDA (Complex) -->
  <g id="traditional-arch">
    <text x="250" y="90" font-family="Arial, sans-serif" font-size="20" font-weight="bold"
          text-anchor="middle" fill="#ef4444">
      Traditional CUDA
    </text>

    <!-- Complex interconnected boxes -->
    <rect x="100" y="110" width="120" height="60" rx="5" fill="#fee2e2" stroke="#ef4444" stroke-width="2"/>
    <text x="160" y="145" font-family="Arial, sans-serif" font-size="12" text-anchor="middle">Thread Mgmt</text>

    <rect x="240" y="110" width="120" height="60" rx="5" fill="#fee2e2" stroke="#ef4444" stroke-width="2"/>
    <text x="300" y="145" font-family="Arial, sans-serif" font-size="12" text-anchor="middle">Block Config</text>

    <rect x="100" y="190" width="120" height="60" rx="5" fill="#fee2e2" stroke="#ef4444" stroke-width="2"/>
    <text x="160" y="225" font-family="Arial, sans-serif" font-size="12" text-anchor="middle">Sync Logic</text>

    <rect x="240" y="190" width="120" height="60" rx="5" fill="#fee2e2" stroke="#ef4444" stroke-width="2"/>
    <text x="300" y="225" font-family="Arial, sans-serif" font-size="12" text-anchor="middle">Shared Mem</text>

    <rect x="170" y="270" width="120" height="60" rx="5" fill="#fee2e2" stroke="#ef4444" stroke-width="2"/>
    <text x="230" y="305" font-family="Arial, sans-serif" font-size="12" text-anchor="middle">Manual Opt</text>

    <!-- Complex connections -->
    <line x1="160" y1="170" x2="160" y2="190" stroke="#dc2626" stroke-width="2"/>
    <line x1="300" y1="170" x2="300" y2="190" stroke="#dc2626" stroke-width="2"/>
    <line x1="160" y1="250" x2="205" y2="270" stroke="#dc2626" stroke-width="2"/>
    <line x1="300" y1="250" x2="255" y2="270" stroke="#dc2626" stroke-width="2"/>
    <line x1="220" y1="140" x2="240" y2="140" stroke="#dc2626" stroke-width="2"/>
    <line x1="220" y1="220" x2="240" y2="220" stroke="#dc2626" stroke-width="2"/>

    <!-- Complexity label -->
    <text x="210" y="365" font-family="Arial, sans-serif" font-size="16" font-weight="bold"
          text-anchor="middle" fill="#991b1b">
      Complex & Error-Prone
    </text>
  </g>

  <!-- Arrow -->
  <g id="simplify-arrow">
    <path d="M 400 250 L 560 250 L 560 230 L 600 260 L 560 290 L 560 270 L 400 270 Z"
          fill="#10b981" stroke="#059669" stroke-width="2"/>
    <text x="500" y="220" font-family="Arial, sans-serif" font-size="18" font-weight="bold"
          text-anchor="middle" fill="#059669">
      Simplify
    </text>
  </g>

  <!-- Tile Programming (Simple) -->
  <g id="tile-arch">
    <text x="750" y="90" font-family="Arial, sans-serif" font-size="20" font-weight="bold"
          text-anchor="middle" fill="#10b981">
      Tile Programming
    </text>

    <!-- Single unified box -->
    <rect x="640" y="140" width="220" height="150" rx="10" fill="#d1fae5" stroke="#10b981" stroke-width="3" filter="url(#shadow)"/>
    <text x="750" y="180" font-family="Arial, sans-serif" font-size="16" font-weight="bold" text-anchor="middle" fill="#065f46">
      Declarative Operations
    </text>
    <text x="750" y="210" font-family="Arial, sans-serif" font-size="13" text-anchor="middle" fill="#047857">
      ct.load()
    </text>
    <text x="750" y="235" font-family="Arial, sans-serif" font-size="13" text-anchor="middle" fill="#047857">
      ct.sum()
    </text>
    <text x="750" y="260" font-family="Arial, sans-serif" font-size="13" text-anchor="middle" fill="#047857">
      ct.store()
    </text>

    <!-- Compiler handles everything -->
    <text x="750" y="325" font-family="Arial, sans-serif" font-size="14" font-style="italic"
          text-anchor="middle" fill="#059669">
      Compiler handles:
    </text>
    <text x="750" y="345" font-family="Arial, sans-serif" font-size="12"
          text-anchor="middle" fill="#047857">
      threads • sync • optimization
    </text>

    <!-- Simplicity label -->
    <text x="750" y="385" font-family="Arial, sans-serif" font-size="16" font-weight="bold"
          text-anchor="middle" fill="#065f46">
      Simple & Reliable
    </text>
  </g>

  <!-- Bottom comparison -->
  <g id="comparison">
    <rect x="100" y="420" width="800" height="60" fill="#6366f1" opacity="0.1" rx="10"/>

    <text x="200" y="445" font-family="Arial, sans-serif" font-size="14" fill="#dc2626">
      Traditional: ❌ 5+ components
    </text>
    <text x="200" y="465" font-family="Arial, sans-serif" font-size="14" fill="#dc2626">
      ❌ Complex interconnections
    </text>

    <text x="600" y="445" font-family="Arial, sans-serif" font-size="14" fill="#059669">
      Tile: ✅ 1 unified component
    </text>
    <text x="600" y="465" font-family="Arial, sans-serif" font-size="14" fill="#059669">
      ✅ Compiler-optimized
    </text>
  </g>
</svg>'''

    return svg


def main():
    """Generate both comparison SVGs."""
    output_dir = Path("docs/assets")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating benefit comparison SVGs...")

    # Code comparison
    code_svg = create_code_comparison_svg()
    code_path = output_dir / "code_comparison.svg"
    with open(code_path, 'w') as f:
        f.write(code_svg)
    print(f"  ✅ {code_path}")

    # Architecture simplification
    arch_svg = create_architecture_simplification_svg()
    arch_path = output_dir / "architecture_simplification.svg"
    with open(arch_path, 'w') as f:
        f.write(arch_svg)
    print(f"  ✅ {arch_path}")

    print("\n✅ Benefit comparison SVGs created!")


if __name__ == "__main__":
    main()
