#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Profile cutileGPT with Nsight Compute

set -e

echo "========================================"
echo "cutileGPT Nsight Compute Profiling"
echo "========================================"

# Output directory
OUTPUT_DIR="profiling_results"
mkdir -p "$OUTPUT_DIR"

# Profile with ncu
echo ""
echo "Running ncu profile..."
echo "Note: This may take several minutes..."
echo ""

ncu \
    --set full \
    --export "$OUTPUT_DIR/cutile_ncu" \
    --force-overwrite \
    --target-processes all \
    --kernel-name-base demangled \
    python profile_performance.py --tool ncu

echo ""
echo "========================================"
echo "Profiling complete!"
echo "========================================"
echo ""
echo "View results with:"
echo "  ncu-ui $OUTPUT_DIR/cutile_ncu.ncu-rep"
echo ""
echo "Or generate report:"
echo "  ncu --import $OUTPUT_DIR/cutile_ncu.ncu-rep"
echo ""
