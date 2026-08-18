from manim import *


class CutileGPTIntro(Scene):
    """cutileGPT 프로젝트 소개 애니메이션"""

    def construct(self):
        # ============================
        # Scene 1: Title
        # ============================
        title = Text("cutileGPT", font_size=72, color=BLUE)
        subtitle = Text(
            "Think in WHAT, not HOW",
            font_size=36,
            color=GREY_B,
        ).next_to(title, DOWN, buff=0.5)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.3), run_time=1)
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ============================
        # Scene 2: The Problem — Traditional CUDA
        # ============================
        problem_title = Text("Traditional CUDA", font_size=48, color=RED_C)
        problem_title.to_edge(UP, buff=0.5)
        self.play(Write(problem_title))

        cuda_code_lines = [
            "// Manual thread management",
            "int tid = threadIdx.x + blockIdx.x * blockDim.x;",
            "int stride = blockDim.x * gridDim.x;",
            "__shared__ float smem[256];",
            "if (tid < N) {",
            "    smem[threadIdx.x] = input[tid];",
            "    __syncthreads();",
            "    output[tid] = gelu(smem[threadIdx.x]);",
            "}",
        ]
        cuda_code = VGroup()
        for line in cuda_code_lines:
            t = Text(line, font_size=18, font="Monospace", color=RED_B)
            cuda_code.add(t)
        cuda_code.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        cuda_code.next_to(problem_title, DOWN, buff=0.6)

        self.play(
            LaggedStart(*[FadeIn(l, shift=RIGHT * 0.3) for l in cuda_code], lag_ratio=0.15),
            run_time=2.5,
        )
        self.wait(0.5)

        # Pain points
        pain_points = VGroup(
            Text("150+ lines of boilerplate", font_size=22, color=RED),
            Text("Manual synchronization", font_size=22, color=RED),
            Text("Thread index arithmetic", font_size=22, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        pain_points.to_edge(DOWN, buff=0.8)

        self.play(
            LaggedStart(*[FadeIn(p) for p in pain_points], lag_ratio=0.3),
            run_time=1.5,
        )
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ============================
        # Scene 3: The Solution — Tile Philosophy
        # ============================
        sol_title = Text("Tile Programming Philosophy", font_size=48, color=GREEN_C)
        sol_title.to_edge(UP, buff=0.5)
        self.play(Write(sol_title))

        tile_code_lines = [
            "# Declarative — no threads!",
            "tile = ct.load(input_ptr, shape)",
            "result = ct.multiply(tile, 0.5)",
            "result = ct.multiply(result, 1 + ct.erf(...))",
            "ct.store(output_ptr, result)",
        ]
        tile_code = VGroup()
        for line in tile_code_lines:
            t = Text(line, font_size=20, font="Monospace", color=GREEN_B)
            tile_code.add(t)
        tile_code.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        tile_code.next_to(sol_title, DOWN, buff=0.6)

        self.play(
            LaggedStart(*[FadeIn(l, shift=RIGHT * 0.3) for l in tile_code], lag_ratio=0.2),
            run_time=2,
        )
        self.wait(0.5)

        benefits = VGroup(
            Text("Only 20 lines", font_size=22, color=GREEN),
            Text("Zero thread management", font_size=22, color=GREEN),
            Text("Compiler handles optimization", font_size=22, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        benefits.to_edge(DOWN, buff=0.8)

        self.play(
            LaggedStart(*[FadeIn(b) for b in benefits], lag_ratio=0.3),
            run_time=1.5,
        )
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ============================
        # Scene 4: Code Reduction — 87%
        # ============================
        section_title = Text("87% Less Code", font_size=48, color=YELLOW)
        section_title.to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Bar chart comparison
        bar_cuda = Rectangle(
            width=1.5, height=4.5, color=RED, fill_opacity=0.7
        )
        bar_tile = Rectangle(
            width=1.5, height=0.6, color=GREEN, fill_opacity=0.7
        )

        bar_cuda.move_to(LEFT * 2 + DOWN * 0.3)
        bar_tile.move_to(RIGHT * 2 + DOWN * 0.3)
        # Align bottoms
        bar_tile.align_to(bar_cuda, DOWN)

        label_cuda = Text("CUDA\n150 lines", font_size=20, color=RED_B).next_to(bar_cuda, DOWN, buff=0.2)
        label_tile = Text("Tile\n20 lines", font_size=20, color=GREEN_B).next_to(bar_tile, DOWN, buff=0.2)

        self.play(GrowFromEdge(bar_cuda, DOWN), FadeIn(label_cuda), run_time=1)
        self.play(GrowFromEdge(bar_tile, DOWN), FadeIn(label_tile), run_time=1)

        arrow = Arrow(bar_cuda.get_right() + UP * 0.5, bar_tile.get_left() + UP * 0.5, color=YELLOW)
        reduction_text = Text("87% reduction", font_size=24, color=YELLOW).next_to(arrow, UP, buff=0.15)
        self.play(GrowArrow(arrow), Write(reduction_text))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ============================
        # Scene 5: Performance — 8.3x speedup
        # ============================
        perf_title = Text("Performance", font_size=48, color=BLUE_C)
        perf_title.to_edge(UP, buff=0.5)
        self.play(Write(perf_title))

        # Speed bars (horizontal)
        cupy_bar = Rectangle(width=5, height=0.6, color=RED, fill_opacity=0.6)
        tile_bar = Rectangle(width=0.6, height=0.6, color=GREEN, fill_opacity=0.8)
        cupy_bar.move_to(UP * 0.5)
        tile_bar.move_to(DOWN * 0.5)
        cupy_bar.align_to(LEFT * 3.5, LEFT)
        tile_bar.align_to(LEFT * 3.5, LEFT)

        cupy_label = Text("CuPy: 5.0 ms", font_size=20, color=RED_B).next_to(cupy_bar, RIGHT, buff=0.3)
        tile_label = Text("Tile: 0.6 ms", font_size=20, color=GREEN_B).next_to(tile_bar, RIGHT, buff=0.3)

        self.play(GrowFromEdge(cupy_bar, LEFT), FadeIn(cupy_label), run_time=1)
        self.play(GrowFromEdge(tile_bar, LEFT), FadeIn(tile_label), run_time=1)

        speedup = Text("8.3x Faster", font_size=40, color=YELLOW).next_to(
            VGroup(cupy_bar, tile_bar), DOWN, buff=0.8
        )
        self.play(Write(speedup))
        self.wait(0.5)

        footnote = Text(
            "GELU kernel — tensor shape: 32 × 512 × 768",
            font_size=16,
            color=GREY,
        ).next_to(speedup, DOWN, buff=0.3)
        self.play(FadeIn(footnote))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ============================
        # Scene 6: Architecture overview
        # ============================
        arch_title = Text("Architecture", font_size=48, color=BLUE_C)
        arch_title.to_edge(UP, buff=0.5)
        self.play(Write(arch_title))

        layers = [
            ("GPT Model", BLUE),
            ("Tile Kernels", GREEN),
            ("CUDA Tile Compiler", ORANGE),
            ("NVIDIA GPU (Tensor Cores)", PURPLE),
        ]
        boxes = VGroup()
        for text, color in layers:
            box = VGroup(
                RoundedRectangle(
                    width=5.5, height=0.8, corner_radius=0.15,
                    color=color, fill_opacity=0.3, stroke_width=2,
                ),
                Text(text, font_size=24, color=color),
            )
            box[1].move_to(box[0])
            boxes.add(box)
        boxes.arrange(DOWN, buff=0.35)
        boxes.next_to(arch_title, DOWN, buff=0.6)

        for i, box in enumerate(boxes):
            self.play(FadeIn(box, shift=DOWN * 0.3), run_time=0.5)

        # Arrows between layers
        for i in range(len(boxes) - 1):
            a = Arrow(
                boxes[i].get_bottom(),
                boxes[i + 1].get_top(),
                buff=0.08,
                color=GREY_B,
                stroke_width=2,
            )
            self.play(GrowArrow(a), run_time=0.3)

        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ============================
        # Scene 7: Key components
        # ============================
        comp_title = Text("Declarative Tile Kernels", font_size=44, color=GREEN_C)
        comp_title.to_edge(UP, buff=0.5)
        self.play(Write(comp_title))

        kernel_data = [
            ("LayerNorm", "Welford's Algorithm"),
            ("GELU", "8.3x Speedup"),
            ("Linear", "Tensor Core MMA"),
            ("Attention", "Flash Attention O(N)"),
            ("Embedding", "Token Lookup"),
        ]
        cards = VGroup()
        for name, desc in kernel_data:
            card = VGroup(
                RoundedRectangle(
                    width=2.2, height=1.4, corner_radius=0.1,
                    color=GREEN, fill_opacity=0.15, stroke_width=1.5,
                ),
                Text(name, font_size=20, color=GREEN_B),
                Text(desc, font_size=14, color=GREY_B),
            )
            card[1].move_to(card[0].get_center() + UP * 0.2)
            card[2].move_to(card[0].get_center() + DOWN * 0.25)
            cards.add(card)

        cards.arrange_in_grid(rows=2, cols=3, buff=0.3)
        cards.next_to(comp_title, DOWN, buff=0.5)

        self.play(
            LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cards], lag_ratio=0.15),
            run_time=2,
        )
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ============================
        # Scene 8: Footprint comparison
        # ============================
        foot_title = Text("200x Smaller Footprint", font_size=44, color=TEAL)
        foot_title.to_edge(UP, buff=0.5)
        self.play(Write(foot_title))

        pytorch_circle = Circle(radius=2, color=RED, fill_opacity=0.3)
        pytorch_label = Text("PyTorch\n~2 GB", font_size=24, color=RED_B).move_to(pytorch_circle)

        cutile_circle = Circle(radius=0.15, color=GREEN, fill_opacity=0.6)
        cutile_label = Text("cutileGPT\n~10 MB", font_size=18, color=GREEN_B)

        pytorch_circle.move_to(LEFT * 1.5 + DOWN * 0.3)
        pytorch_label.move_to(pytorch_circle)
        cutile_circle.next_to(pytorch_circle, RIGHT, buff=1.5)
        cutile_label.next_to(cutile_circle, DOWN, buff=0.3)

        self.play(Create(pytorch_circle), Write(pytorch_label), run_time=1)
        self.play(Create(cutile_circle), Write(cutile_label), run_time=1)

        edge_text = Text("Edge Deployment Ready", font_size=28, color=TEAL).to_edge(DOWN, buff=0.8)
        self.play(Write(edge_text))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ============================
        # Scene 9: Closing
        # ============================
        closing = Text("cutileGPT", font_size=72, color=BLUE)
        tagline = Text(
            "Declarative GPU Programming\nfor the Future",
            font_size=32,
            color=GREY_B,
        ).next_to(closing, DOWN, buff=0.5)

        highlights = VGroup(
            Text("8.3x Faster Kernels", font_size=24, color=GREEN),
            Text("87% Less Code", font_size=24, color=YELLOW),
            Text("200x Smaller Footprint", font_size=24, color=TEAL),
        ).arrange(RIGHT, buff=1).next_to(tagline, DOWN, buff=0.8)

        github = Text(
            "github.com/falcons-eyes/cutileGPT",
            font_size=20,
            color=GREY,
        ).to_edge(DOWN, buff=0.5)

        self.play(Write(closing), run_time=1.5)
        self.play(FadeIn(tagline, shift=UP * 0.3))
        self.play(
            LaggedStart(*[FadeIn(h, shift=UP * 0.2) for h in highlights], lag_ratio=0.2),
            run_time=1.5,
        )
        self.play(FadeIn(github))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])
