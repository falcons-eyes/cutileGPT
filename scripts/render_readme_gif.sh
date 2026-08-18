#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
render_dir=$(mktemp -d)
output_path="$project_dir/docs/assets/readme/tile-programming.gif"

cleanup() {
    rm -rf "$render_dir"
}
trap cleanup EXIT

uvx --from manim manim \
    -qm \
    --fps 24 \
    -r 960,540 \
    --media_dir "$render_dir" \
    "$project_dir/scripts/readme_tile_animation.py" \
    TileProgrammingValue

video_path=$(find "$render_dir" -type f -name 'TileProgrammingValue.mp4' -print -quit)
if [[ -z "$video_path" ]]; then
    echo "Manim output was not found" >&2
    exit 1
fi

ffmpeg -hide_banner -loglevel error -y \
    -i "$video_path" \
    -filter_complex \
    "fps=12,scale=960:-1:flags=lanczos,split[frames][palette_input];[palette_input]palettegen=max_colors=128:stats_mode=diff[palette];[frames][palette]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
    -loop 0 \
    "$output_path"

echo "Wrote ${output_path#"$project_dir/"}"
