#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manim scene for the tile-programming GIF embedded in README.md."""

from __future__ import annotations

from manim import (
    AnimationGroup,
    Arrow,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    LaggedStart,
    Line,
    Rectangle,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    Transform,
    VGroup,
    Write,
)

BG = "#0B1020"
PANEL = "#121A2D"
RAISED = "#19243A"
BORDER = "#2A3855"
TEXT = "#F3F7FC"
MUTED = "#A3B0C4"
CYAN = "#70D9EC"
PURPLE = "#A99AF4"
GREEN = "#64D6A0"
RED = "#F27D92"
AMBER = "#F2B84B"
MONO = "DejaVu Sans Mono"
SANS = "DejaVu Sans"


def make_grid(
    rows: int,
    cols: int,
    *,
    side: float = 0.36,
    color: str = BORDER,
    fill: str = RAISED,
) -> VGroup:
    cells = VGroup(
        *[
            Square(
                side_length=side,
                stroke_color=color,
                stroke_width=1.1,
                fill_color=fill,
                fill_opacity=0.76,
            )
            for _ in range(rows * cols)
        ]
    )
    cells.arrange_in_grid(rows=rows, cols=cols, buff=0.055)
    return cells


def panel(center_x: float, title: str, subtitle: str, accent: str) -> tuple[VGroup, VGroup]:
    card = RoundedRectangle(
        width=6.25,
        height=4.75,
        corner_radius=0.22,
        stroke_color=BORDER,
        stroke_width=1.4,
        fill_color=PANEL,
        fill_opacity=1,
    ).move_to([center_x, -0.2, 0])
    accent_line = Rectangle(
        width=0.07,
        height=0.62,
        stroke_width=0,
        fill_color=accent,
        fill_opacity=1,
    ).move_to([center_x - 2.78, 1.48, 0])
    heading = Text(title, font=SANS, font_size=25, color=TEXT)
    heading.move_to([center_x - 2.47 + heading.width / 2, 1.63, 0])
    subtitle_text = Text(subtitle, font=SANS, font_size=15, color=MUTED)
    subtitle_text.move_to([heading.get_left()[0] + subtitle_text.width / 2, 1.27, 0])
    return VGroup(card, accent_line), VGroup(heading, subtitle_text)


def code_pill(label: str, x: float, y: float, width: float, accent: str) -> VGroup:
    box = RoundedRectangle(
        width=width,
        height=0.48,
        corner_radius=0.12,
        stroke_color=accent,
        stroke_width=1.15,
        fill_color=RAISED,
        fill_opacity=1,
    ).move_to([x, y, 0])
    label_text = Text(label, font=MONO, font_size=14, color=accent).move_to(box)
    return VGroup(box, label_text)


def head_box(label: str, x: float, y: float, color: str, width: float = 1.05) -> VGroup:
    box = RoundedRectangle(
        width=width,
        height=0.62,
        corner_radius=0.13,
        stroke_color=color,
        stroke_width=1.25,
        fill_color=RAISED,
        fill_opacity=1,
    ).move_to([x, y, 0])
    label_text = Text(label, font=MONO, font_size=15, color=color).move_to(box)
    return VGroup(box, label_text)


class TileProgrammingValue(Scene):
    """Show abstraction leverage first, then the MHA-to-GQA architecture change."""

    def construct(self) -> None:
        self.camera.background_color = BG

        kicker = Text(
            "cutileGPT  ·  NVIDIA cuTile Python",
            font=SANS,
            font_size=16,
            color=CYAN,
        ).move_to([-4.78, 3.55, 0])
        title = Text(
            "Think in tiles. Not threads.", font=SANS, font_size=39, color=TEXT
        ).move_to([0, 3.05, 0])
        rule = Rectangle(
            width=13.1,
            height=0.018,
            stroke_width=0,
            fill_color=BORDER,
            fill_opacity=1,
        ).move_to([0, 2.64, 0])
        self.play(FadeIn(kicker), Write(title), FadeIn(rule), run_time=0.65)

        left_panel, left_heading = panel(-3.42, "Manual CUDA", "Describe HOW", RED)
        right_panel, right_heading = panel(3.42, "Tile Programming", "Describe WHAT", CYAN)
        self.play(
            FadeIn(left_panel),
            FadeIn(right_panel),
            FadeIn(left_heading),
            FadeIn(right_heading),
            run_time=0.55,
        )

        left_grid = make_grid(5, 8, side=0.31).move_to([-3.42, 0.10, 0])
        right_grid = make_grid(5, 8, side=0.31).move_to([3.42, 0.10, 0])
        tile_border = RoundedRectangle(
            width=3.05,
            height=1.96,
            corner_radius=0.15,
            stroke_color=CYAN,
            stroke_width=3,
            fill_opacity=0,
        ).move_to(right_grid)
        self.play(FadeIn(left_grid), FadeIn(right_grid), Create(tile_border), run_time=0.45)

        thread_xs = [-5.55, -4.92, -4.29, -3.66, -3.03, -2.40, -1.77, -1.14]
        thread_dots = VGroup(
            *[Dot([x, -1.15, 0], radius=0.075, color=RED) for x in thread_xs]
        )
        thread_arrows = VGroup()
        target_indices = [0, 9, 18, 27, 36, 13, 22, 31]
        for dot, target_index in zip(thread_dots, target_indices, strict=True):
            thread_arrows.add(
                Arrow(
                    dot.get_top(),
                    left_grid[target_index].get_bottom(),
                    buff=0.045,
                    stroke_width=1.15,
                    color=RED,
                    max_tip_length_to_length_ratio=0.12,
                )
            )
        thread_label = Text("threadIdx.x", font=MONO, font_size=13, color=RED).move_to(
            [-3.42, -1.46, 0]
        )

        load = code_pill("ct.load", 1.82, -1.12, 1.35, CYAN)
        mma = code_pill("ct.mma", 3.42, -1.12, 1.35, PURPLE)
        store = code_pill("ct.store", 5.02, -1.12, 1.35, GREEN)
        pipeline_arrows = VGroup(
            Arrow(load.get_right(), mma.get_left(), buff=0.08, color=MUTED, stroke_width=1.3),
            Arrow(mma.get_right(), store.get_left(), buff=0.08, color=MUTED, stroke_width=1.3),
        )
        self.play(
            LaggedStart(*[FadeIn(dot) for dot in thread_dots], lag_ratio=0.07),
            LaggedStart(*[Create(arrow) for arrow in thread_arrows], lag_ratio=0.04),
            FadeIn(thread_label),
            FadeIn(load),
            FadeIn(mma),
            FadeIn(store),
            FadeIn(pipeline_arrows),
            run_time=0.8,
        )

        cuda_tags = VGroup(
            code_pill("blockIdx", -5.16, -1.95, 1.28, RED),
            code_pill("__shared__", -3.42, -1.95, 1.58, RED),
            code_pill("__syncthreads()", -1.64, -1.95, 2.08, RED),
        )
        self.play(FadeIn(cuda_tags), run_time=0.35)

        left_cells = [left_grid[index] for index in target_indices]
        self.play(
            LaggedStart(
                *[
                    cell.animate.set_fill(RED, opacity=0.78).set_stroke(RED, width=1.6)
                    for cell in left_cells
                ],
                lag_ratio=0.08,
            ),
            AnimationGroup(
                *[
                    cell.animate.set_fill(CYAN, opacity=0.78).set_stroke(CYAN, width=1.2)
                    for cell in right_grid
                ],
                lag_ratio=0,
            ),
            load[0].animate.set_fill(CYAN, opacity=0.2),
            run_time=0.7,
        )
        self.play(
            AnimationGroup(
                *[
                    cell.animate.set_fill(PURPLE, opacity=0.78).set_stroke(PURPLE, width=1.2)
                    for cell in right_grid
                ],
                lag_ratio=0,
            ),
            load[0].animate.set_fill(RAISED, opacity=1),
            mma[0].animate.set_fill(PURPLE, opacity=0.2),
            tile_border.animate.set_stroke(PURPLE),
            run_time=0.42,
        )
        self.play(
            AnimationGroup(
                *[
                    cell.animate.set_fill(GREEN, opacity=0.78).set_stroke(GREEN, width=1.2)
                    for cell in right_grid
                ],
                lag_ratio=0,
            ),
            mma[0].animate.set_fill(RAISED, opacity=1),
            store[0].animate.set_fill(GREEN, opacity=0.2),
            tile_border.animate.set_stroke(GREEN),
            run_time=0.42,
        )

        compiler_caption = Text(
            "Compiler owns thread mapping  ·  memory movement  ·  synchronization",
            font=SANS,
            font_size=18,
            color=GREEN,
        ).move_to([0, -3.16, 0])
        self.play(FadeIn(compiler_caption), run_time=0.35)
        self.wait(0.35)

        stage_one = VGroup(
            left_panel,
            left_heading,
            right_panel,
            right_heading,
            left_grid,
            right_grid,
            tile_border,
            thread_dots,
            thread_arrows,
            thread_label,
            load,
            mma,
            store,
            pipeline_arrows,
            cuda_tags,
            compiler_caption,
        )
        next_title = Text(
            "Change the model. Keep the kernel readable.",
            font=SANS,
            font_size=35,
            color=TEXT,
        )
        next_title.move_to(title)
        self.play(FadeOut(stage_one), Transform(title, next_title), run_time=0.5)

        gqa_panel = RoundedRectangle(
            width=12.85,
            height=4.85,
            corner_radius=0.22,
            stroke_color=BORDER,
            stroke_width=1.4,
            fill_color=PANEL,
            fill_opacity=1,
        ).move_to([0, -0.22, 0])
        mode_label = Text(
            "MHA  ·  one KV head per query head",
            font=SANS,
            font_size=19,
            color=MUTED,
        ).move_to([0, 1.73, 0])
        self.play(FadeIn(gqa_panel), FadeIn(mode_label), run_time=0.4)

        head_xs = [-5.05, -3.62, -2.18, -0.73, 0.73, 2.18, 3.62, 5.05]
        q_heads = VGroup(
            *[head_box(f"Q{i}", x, 0.92, CYAN) for i, x in enumerate(head_xs)]
        )
        mha_kv = VGroup(
            *[head_box(f"KV{i}", x, -0.46, PURPLE) for i, x in enumerate(head_xs)]
        )
        mha_lines = VGroup(
            *[
                Line(q.get_bottom(), kv.get_top(), color=BORDER, stroke_width=1.5)
                for q, kv in zip(q_heads, mha_kv, strict=True)
            ]
        )
        self.play(
            LaggedStart(*[GrowFromCenter(head) for head in q_heads], lag_ratio=0.05),
            LaggedStart(*[GrowFromCenter(head) for head in mha_kv], lag_ratio=0.05),
            FadeIn(mha_lines),
            run_time=0.7,
        )

        gqa_label = Text(
            "GQA  ·  four query heads share one KV head",
            font=SANS,
            font_size=19,
            color=GREEN,
        )
        gqa_label.move_to(mode_label)
        grouped_kv = VGroup(
            head_box("KV0", -2.15, -0.46, GREEN, width=1.55),
            head_box("KV1", 2.15, -0.46, GREEN, width=1.55),
        )
        grouped_lines = VGroup()
        for index, q_head in enumerate(q_heads):
            target = grouped_kv[0] if index < 4 else grouped_kv[1]
            grouped_lines.add(
                Line(q_head.get_bottom(), target.get_top(), color=GREEN, stroke_width=1.45)
            )
        self.play(
            FadeOut(mha_kv),
            FadeOut(mha_lines),
            Transform(mode_label, gqa_label),
            FadeIn(grouped_lines),
            GrowFromCenter(grouped_kv[0]),
            GrowFromCenter(grouped_kv[1]),
            run_time=0.65,
        )

        code = code_pill(
            "kv_head = q_head // (N_HEAD // N_KV_HEAD)",
            0,
            -1.67,
            7.2,
            CYAN,
        )
        payoff = Text(
            "one semantic index   ·   4× smaller KV cache",
            font=SANS,
            font_size=20,
            color=GREEN,
        ).move_to([0, -2.25, 0])
        self.play(FadeIn(code), run_time=0.35)
        self.play(Write(payoff), run_time=0.45)
        self.wait(0.55)

        scene_two = VGroup(
            gqa_panel,
            mode_label,
            q_heads,
            grouped_kv,
            grouped_lines,
            code,
            payoff,
        )
        final_title = Text("cutileGPT", font=SANS, font_size=64, color=CYAN).move_to(
            [0, 0.58, 0]
        )
        final_subtitle = Text(
            "Modern transformers. Readable GPU kernels.",
            font=SANS,
            font_size=27,
            color=TEXT,
        ).move_to([0, -0.18, 0])
        final_features = Text(
            "RMSNorm   ·   RoPE   ·   GQA   ·   SwiGLU   ·   KV cache",
            font=SANS,
            font_size=18,
            color=MUTED,
        ).move_to([0, -0.85, 0])
        self.play(
            FadeOut(scene_two),
            FadeOut(title),
            FadeOut(rule),
            FadeOut(kicker),
            run_time=0.4,
        )
        self.play(FadeIn(final_title), FadeIn(final_subtitle), FadeIn(final_features), run_time=0.55)
        self.wait(0.65)
        self.play(FadeOut(VGroup(final_title, final_subtitle, final_features)), run_time=0.4)
