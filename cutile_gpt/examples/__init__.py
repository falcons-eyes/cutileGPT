# SPDX-License-Identifier: Apache-2.0
"""
Tile Philosophy Examples

Educational implementations demonstrating the Tile Programming philosophy.

These files show:
- How to write declarative tile kernels
- Comparison with traditional CUDA approaches
- No explicit thread management
- Compiler-driven optimization

Files:
- linear_tile.py: Matrix multiplication using tile programming
- attention_tile.py: Causal self-attention using tile programming
- layernorm_tile.py: Layer normalization using tile programming
- gelu_tile.py: GELU activation using tile programming
"""

# These modules are for educational purposes and standalone testing
# Import them directly if needed:
#   from cutile_gpt.examples import linear_tile
