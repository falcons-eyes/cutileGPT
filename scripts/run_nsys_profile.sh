#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Profile cutileGPT with Nsight Systems

set -e

echo "========================================"
echo "cutileGPT Nsight Systems Profiling"
echo "========================================"

# Output directory
OUTPUT_DIR="profiling_results"
mkdir -p "$OUTPUT_DIR"

# Profile with nsys
echo ""
echo "Running nsys profile..."
nsys profile \
    --output="$OUTPUT_DIR/cutile_nsys" \
    --stats=true \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    uv run python profile_performance.py --tool nsys

echo ""
echo "========================================"
echo "Profiling complete!"
echo "========================================"
echo ""
echo "View results with:"
echo "  nsys-ui $OUTPUT_DIR/cutile_nsys.nsys-rep"
echo ""
echo "Or generate report:"
echo "  nsys stats $OUTPUT_DIR/cutile_nsys.nsys-rep"
echo ""
