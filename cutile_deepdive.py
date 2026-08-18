from manim import *
import numpy as np


# ==============================
# Color Palette
# ==============================
C_INPUT = BLUE_C
C_OUTPUT = GREEN_C
C_WEIGHT = ORANGE
C_ACCENT = YELLOW
C_TILE = TEAL
C_ATTN = PURPLE_B
C_RESIDUAL = RED_C
C_GELU = PINK
C_BG_BOX = GREY_E


def make_tensor_grid(rows, cols, cell_size=0.3, color=BLUE, fill_opacity=0.15, label=None):
    """Create a visual tensor grid."""
    cells = VGroup()
    for r in range(rows):
        for c in range(cols):
            sq = Square(side_length=cell_size, color=color, fill_opacity=fill_opacity, stroke_width=1)
            sq.move_to(RIGHT * c * cell_size + DOWN * r * cell_size)
            cells.add(sq)
    cells.move_to(ORIGIN)
    result = VGroup(cells)
    if label:
        lbl = Text(label, font_size=14, color=color).next_to(cells, DOWN, buff=0.15)
        result.add(lbl)
    return result


def make_tensor_block(width, height, color=BLUE, label="", sublabel="", fill_opacity=0.25):
    """Create a rounded rectangle representing a tensor."""
    box = RoundedRectangle(
        width=width, height=height, corner_radius=0.1,
        color=color, fill_opacity=fill_opacity, stroke_width=2,
    )
    parts = VGroup(box)
    if label:
        lbl = Text(label, font_size=18, color=color, weight=BOLD)
        lbl.move_to(box.get_center() + UP * 0.15 if sublabel else ORIGIN)
        parts.add(lbl)
    if sublabel:
        slbl = Text(sublabel, font_size=13, color=GREY_B)
        slbl.move_to(box.get_center() + DOWN * 0.2)
        parts.add(slbl)
    return parts


def data_flow_arrow(start, end, color=WHITE):
    return Arrow(start, end, buff=0.08, color=color, stroke_width=2, max_tip_length_to_length_ratio=0.15)


# ====================================================================
# Scene 1: Title + High-Level Tensor Flow
# ====================================================================
class S01_TitleAndOverview(Scene):
    def construct(self):
        title = Text("cutileGPT", font_size=64, color=C_INPUT)
        subtitle = Text("Deep Dive: Tensor Data Flow", font_size=28, color=GREY_B)
        subtitle.next_to(title, DOWN, buff=0.4)
        self.play(Write(title), run_time=1)
        self.play(FadeIn(subtitle, shift=UP * 0.2))
        self.wait(1)

        # High-level pipeline
        self.play(FadeOut(title), FadeOut(subtitle))

        stages = [
            ("Tokens\n(2, 64)", C_INPUT),
            ("Embed\n(2,64,768)", C_TILE),
            ("12x Blocks\n(2,64,768)", C_ATTN),
            ("LayerNorm\n(2,64,768)", C_ACCENT),
            ("Logits\n(2,64,50257)", C_OUTPUT),
        ]
        boxes = VGroup()
        for txt, col in stages:
            b = make_tensor_block(2.2, 1.0, color=col, label=txt)
            boxes.add(b)
        boxes.arrange(RIGHT, buff=0.6)
        boxes.scale(0.85).move_to(ORIGIN)

        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.3) for b in boxes], lag_ratio=0.2), run_time=2)

        arrows = VGroup()
        for i in range(len(boxes) - 1):
            a = data_flow_arrow(boxes[i].get_right(), boxes[i + 1].get_left())
            arrows.add(a)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.15), run_time=1.5)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ====================================================================
# Scene 2: Embedding — Token IDs to Dense Vectors
# ====================================================================
class S02_Embedding(Scene):
    def construct(self):
        title = Text("Embedding: Tokens to Vectors", font_size=36, color=C_TILE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Token IDs as colored squares
        token_ids = [42, 1024, 305, 7, 999]
        tok_group = VGroup()
        for tid in token_ids:
            sq = Square(side_length=0.5, color=C_INPUT, fill_opacity=0.4, stroke_width=1.5)
            lbl = Text(str(tid), font_size=14, color=WHITE)
            lbl.move_to(sq)
            tok_group.add(VGroup(sq, lbl))
        tok_group.arrange(RIGHT, buff=0.15)
        tok_label = Text("Token IDs (5,)", font_size=16, color=C_INPUT)
        tok_label.next_to(tok_group, DOWN, buff=0.2)
        tok_all = VGroup(tok_group, tok_label).move_to(LEFT * 4 + DOWN * 0.3)
        self.play(FadeIn(tok_all))

        # Embedding matrix
        emb_grid = make_tensor_grid(8, 6, cell_size=0.22, color=C_WEIGHT, fill_opacity=0.1, label="Weights (50257, 768)")
        emb_grid.move_to(ORIGIN + DOWN * 0.3)
        self.play(FadeIn(emb_grid))

        # Arrows from tokens to rows
        gather_arrows = VGroup()
        for i, tok in enumerate(tok_group):
            a = Arrow(
                tok.get_right(), emb_grid[0][i * 6].get_left(),
                buff=0.05, color=C_ACCENT, stroke_width=1.5,
                max_tip_length_to_length_ratio=0.2,
            )
            gather_arrows.add(a)
        self.play(LaggedStart(*[GrowArrow(a) for a in gather_arrows], lag_ratio=0.1), run_time=1)

        # Highlight gathered rows
        row_highlights = VGroup()
        for i in range(5):
            cells = VGroup(*[emb_grid[0][i * 6 + c] for c in range(6)])
            hl = SurroundingRectangle(cells, color=C_ACCENT, buff=0.02, stroke_width=2)
            row_highlights.add(hl)
        self.play(LaggedStart(*[Create(h) for h in row_highlights], lag_ratio=0.1), run_time=1)

        # Output vectors
        out_grid = make_tensor_grid(5, 6, cell_size=0.22, color=C_OUTPUT, fill_opacity=0.25, label="Embedded (5, 768)")
        out_grid.move_to(RIGHT * 4 + DOWN * 0.3)
        out_arrows = VGroup()
        for i in range(5):
            a = Arrow(
                emb_grid[0][(i + 1) * 6 - 1].get_right(), out_grid[0][i * 6].get_left(),
                buff=0.05, color=C_OUTPUT, stroke_width=1.5,
                max_tip_length_to_length_ratio=0.2,
            )
            out_arrows.add(a)
        self.play(LaggedStart(*[GrowArrow(a) for a in out_arrows], lag_ratio=0.1), run_time=1)
        self.play(FadeIn(out_grid))

        # Position embedding addition
        plus_sign = Text("+", font_size=36, color=C_ACCENT).next_to(out_grid, DOWN, buff=0.3)
        pos_label = Text("Position Emb (64, 768)", font_size=16, color=C_ACCENT).next_to(plus_sign, RIGHT, buff=0.2)
        self.play(FadeIn(plus_sign), FadeIn(pos_label))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ====================================================================
# Scene 3: Tiled Matrix Multiplication (Linear Kernel)
# ====================================================================
class S03_TiledMatmul(Scene):
    def construct(self):
        title = Text("Tiled Matrix Multiply", font_size=36, color=C_WEIGHT)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Matrix A
        A = make_tensor_grid(6, 8, cell_size=0.28, color=C_INPUT, fill_opacity=0.12)
        A_label = Text("A (128, 768)", font_size=14, color=C_INPUT).next_to(A[0], DOWN, buff=0.12)
        A_all = VGroup(A, A_label).move_to(LEFT * 4.5 + DOWN * 0.5)

        # Matrix B
        B = make_tensor_grid(8, 6, cell_size=0.28, color=C_WEIGHT, fill_opacity=0.12)
        B_label = Text("B (768, 3072)", font_size=14, color=C_WEIGHT).next_to(B[0], DOWN, buff=0.12)
        B_all = VGroup(B, B_label).move_to(LEFT * 0.5 + DOWN * 0.5)

        # Output C
        C = make_tensor_grid(6, 6, cell_size=0.28, color=C_OUTPUT, fill_opacity=0.08)
        C_label = Text("C (128, 3072)", font_size=14, color=C_OUTPUT).next_to(C[0], DOWN, buff=0.12)
        C_all = VGroup(C, C_label).move_to(RIGHT * 4 + DOWN * 0.5)

        self.play(FadeIn(A_all), FadeIn(B_all), FadeIn(C_all))

        # Show tile traversal
        # Highlight a tile block in A, B, and C
        tile_a = SurroundingRectangle(
            VGroup(A[0][0], A[0][1], A[0][8], A[0][9]),
            color=C_ACCENT, buff=0.02, stroke_width=3,
        )
        tile_b = SurroundingRectangle(
            VGroup(B[0][0], B[0][1], B[0][6], B[0][7]),
            color=C_ACCENT, buff=0.02, stroke_width=3,
        )
        tile_c = SurroundingRectangle(
            VGroup(C[0][0], C[0][1], C[0][6], C[0][7]),
            color=C_ACCENT, buff=0.02, stroke_width=3,
        )

        tile_a_lbl = Text("Tile (32,32)", font_size=12, color=C_ACCENT).next_to(tile_a, UP, buff=0.08)
        tile_b_lbl = Text("Tile (32,32)", font_size=12, color=C_ACCENT).next_to(tile_b, UP, buff=0.08)

        self.play(Create(tile_a), Create(tile_b), Create(tile_c))
        self.play(FadeIn(tile_a_lbl), FadeIn(tile_b_lbl))
        self.wait(0.5)

        # K-loop animation: slide tile highlight across K dimension of A and down K dimension of B
        ops_text = Text("acc += ct.mma(a_tile, b_tile, acc)", font_size=16, font="Monospace", color=C_ACCENT)
        ops_text.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(ops_text))

        # Animate tiles sliding along K axis (simulate 3 steps)
        for step in range(3):
            col_offset_a = (step + 1) * 2
            row_offset_b = (step + 1) * 2
            if col_offset_a + 1 < 8 and row_offset_b + 1 < 8:
                new_tile_a = SurroundingRectangle(
                    VGroup(A[0][col_offset_a], A[0][col_offset_a + 1],
                           A[0][8 + col_offset_a], A[0][8 + col_offset_a + 1]),
                    color=C_ACCENT, buff=0.02, stroke_width=3,
                )
                new_tile_b = SurroundingRectangle(
                    VGroup(B[0][row_offset_b * 6], B[0][row_offset_b * 6 + 1],
                           B[0][(row_offset_b + 1) * 6], B[0][(row_offset_b + 1) * 6 + 1]),
                    color=C_ACCENT, buff=0.02, stroke_width=3,
                )
                self.play(
                    Transform(tile_a, new_tile_a),
                    Transform(tile_b, new_tile_b),
                    run_time=0.5,
                )
                # Flash the output tile
                self.play(tile_c.animate.set_stroke(width=5), run_time=0.15)
                self.play(tile_c.animate.set_stroke(width=3), run_time=0.15)

        # Swizzle explanation
        swizzle_text = Text("2D Swizzle: L2 Cache-Friendly Block Order", font_size=16, color=GREY_B)
        swizzle_text.next_to(ops_text, DOWN, buff=0.2)
        self.play(FadeIn(swizzle_text))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ====================================================================
# Scene 4: GELU Activation — Element-wise Tensor Transform
# ====================================================================
class S04_GELU(Scene):
    def construct(self):
        title = Text("GELU Activation", font_size=36, color=C_GELU)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Plot GELU curve
        axes = Axes(
            x_range=[-4, 4, 1], y_range=[-1, 4, 1],
            x_length=5, y_length=3,
            axis_config={"include_numbers": True, "font_size": 18},
        ).move_to(LEFT * 3 + DOWN * 0.5)
        axes_label = Text("GELU(x) = 0.5x(1 + tanh(...))", font_size=14, color=C_GELU)
        axes_label.next_to(axes, DOWN, buff=0.2)

        relu_graph = axes.plot(lambda x: max(0, x), x_range=[-4, 4], color=GREY, stroke_width=1.5)
        relu_label = Text("ReLU", font_size=12, color=GREY).move_to(axes.c2p(2.5, 3))

        gelu_graph = axes.plot(
            lambda x: 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))),
            x_range=[-4, 4], color=C_GELU, stroke_width=3,
        )
        gelu_label = Text("GELU", font_size=14, color=C_GELU).move_to(axes.c2p(2.5, 2.2))

        self.play(Create(axes), FadeIn(axes_label))
        self.play(Create(relu_graph), FadeIn(relu_label), run_time=0.8)
        self.play(Create(gelu_graph), FadeIn(gelu_label), run_time=1.2)

        # Tensor tile processing visualization on the right
        tile_title = Text("Tile Processing (32 x 128)", font_size=16, color=C_GELU)
        tile_title.move_to(RIGHT * 3.5 + UP * 1.5)
        self.play(FadeIn(tile_title))

        # Input tile grid
        in_tile = make_tensor_grid(4, 8, cell_size=0.25, color=C_INPUT, fill_opacity=0.15)
        in_tile[0].move_to(RIGHT * 3.5)
        in_lbl = Text("Input Tile", font_size=12, color=C_INPUT).next_to(in_tile[0], UP, buff=0.1)

        self.play(FadeIn(in_tile[0]), FadeIn(in_lbl))

        # Wave animation: highlight cells in sequence to show element-wise
        wave_cells = VGroup()
        for i in range(32):
            cell = in_tile[0][i]
            wave_cells.add(cell)

        self.play(
            LaggedStart(
                *[cell.animate.set_fill(C_GELU, opacity=0.5) for cell in wave_cells],
                lag_ratio=0.03,
            ),
            run_time=1.5,
        )

        # Output tile
        out_tile = make_tensor_grid(4, 8, cell_size=0.25, color=C_GELU, fill_opacity=0.3)
        out_tile[0].move_to(RIGHT * 3.5 + DOWN * 2)
        out_lbl = Text("GELU Output", font_size=12, color=C_GELU).next_to(out_tile[0], DOWN, buff=0.1)

        transform_arrow = Arrow(in_tile[0].get_bottom(), out_tile[0].get_top(), buff=0.1, color=C_GELU)
        self.play(GrowArrow(transform_arrow))
        self.play(FadeIn(out_tile[0]), FadeIn(out_lbl))

        # 8.3x speedup callout
        speedup = Text("8.3x faster than CuPy", font_size=20, color=C_ACCENT)
        speedup.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(speedup))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ====================================================================
# Scene 5: LayerNorm — Two-Pass Row Normalization
# ====================================================================
class S05_LayerNorm(Scene):
    def construct(self):
        title = Text("LayerNorm: Two-Pass Normalization", font_size=34, color=C_ACCENT)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Input tensor: show as row-major grid
        rows, cols = 5, 10
        grid = VGroup()
        for r in range(rows):
            for c in range(cols):
                val = np.random.uniform(-2, 2)
                intensity = (val + 2) / 4  # normalize to 0-1
                sq = Square(
                    side_length=0.35, stroke_width=1, color=C_INPUT,
                    fill_opacity=intensity * 0.6,
                )
                sq.move_to(RIGHT * c * 0.38 + DOWN * r * 0.38)
                grid.add(sq)
        grid.move_to(LEFT * 2.5 + DOWN * 0.3)
        grid_label = Text("Input (128, 768) — each row normalized", font_size=14, color=C_INPUT)
        grid_label.next_to(grid, DOWN, buff=0.2)
        self.play(FadeIn(grid), FadeIn(grid_label))

        # Pass 1: highlight row by row for mean/var
        pass1_label = Text("Pass 1: Compute Mean & Variance", font_size=18, color=C_ACCENT)
        pass1_label.move_to(RIGHT * 3 + UP * 1)
        self.play(FadeIn(pass1_label))

        for r in range(rows):
            row_cells = VGroup(*[grid[r * cols + c] for c in range(cols)])
            highlight = SurroundingRectangle(row_cells, color=C_ACCENT, buff=0.02, stroke_width=2)
            if r == 0:
                self.play(Create(highlight), run_time=0.4)
            else:
                new_hl = SurroundingRectangle(row_cells, color=C_ACCENT, buff=0.02, stroke_width=2)
                self.play(Transform(highlight, new_hl), run_time=0.25)

        # Show mean/var symbols
        stats = MathTex(r"\mu = \frac{1}{N}\sum x_i", r"\quad", r"\sigma^2 = \frac{1}{N}\sum (x_i - \mu)^2", font_size=28)
        stats.move_to(RIGHT * 3 + DOWN * 0.2)
        stats[0].set_color(C_ACCENT)
        stats[2].set_color(C_ACCENT)
        self.play(Write(stats), run_time=1.5)
        self.wait(0.5)

        # Pass 2
        pass2_label = Text("Pass 2: Normalize + Affine", font_size=18, color=C_OUTPUT)
        pass2_label.move_to(RIGHT * 3 + DOWN * 1.2)
        formula = MathTex(r"y = \gamma \cdot \frac{x - \mu}{\sigma} + \beta", font_size=28, color=C_OUTPUT)
        formula.next_to(pass2_label, DOWN, buff=0.15)
        self.play(FadeIn(pass2_label), Write(formula))

        # Transform grid to normalized (uniform opacity)
        anims = []
        for sq in grid:
            anims.append(sq.animate.set_fill(C_OUTPUT, opacity=0.3))
        self.play(*anims, run_time=1.5)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ====================================================================
# Scene 6: Multi-Head Attention (Flash Attention)
# ====================================================================
class S06_Attention(Scene):
    def construct(self):
        title = Text("Flash Attention: Online Softmax", font_size=34, color=C_ATTN)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Q, K, V tensors
        q_block = make_tensor_block(1.4, 1.0, C_INPUT, "Q", "(64, 64)")
        k_block = make_tensor_block(1.4, 1.0, C_WEIGHT, "K", "(64, 64)")
        v_block = make_tensor_block(1.4, 1.0, C_OUTPUT, "V", "(64, 64)")

        qkv = VGroup(q_block, k_block, v_block).arrange(RIGHT, buff=0.4)
        qkv.move_to(UP * 1.5 + LEFT * 3)
        head_label = Text("Per Head (12 heads parallel)", font_size=14, color=GREY_B)
        head_label.next_to(qkv, UP, buff=0.15)
        self.play(FadeIn(qkv), FadeIn(head_label))

        # Step 1: Q @ K^T
        qk_block = make_tensor_block(1.6, 1.0, C_ATTN, "Q @ K^T", "(64, 64)")
        qk_block.move_to(RIGHT * 1 + UP * 1.5)

        a1 = data_flow_arrow(q_block.get_right(), qk_block.get_left(), C_INPUT)
        a2 = data_flow_arrow(k_block.get_right(), qk_block.get_left() + DOWN * 0.15, C_WEIGHT)
        self.play(GrowArrow(a1), GrowArrow(a2), FadeIn(qk_block), run_time=0.8)

        # Causal mask visualization
        mask_size = 6
        mask_grid = VGroup()
        for r in range(mask_size):
            for c in range(mask_size):
                sq = Square(side_length=0.25, stroke_width=0.5)
                if c <= r:
                    sq.set_fill(C_ATTN, opacity=0.5)
                    sq.set_stroke(C_ATTN)
                else:
                    sq.set_fill(RED, opacity=0.15)
                    sq.set_stroke(RED, width=0.5)
                sq.move_to(RIGHT * c * 0.27 + DOWN * r * 0.27)
                mask_grid.add(sq)
        mask_grid.move_to(RIGHT * 4 + UP * 1.5)
        mask_label = Text("Causal Mask", font_size=13, color=C_ATTN).next_to(mask_grid, DOWN, buff=0.1)
        neg_inf = Text("-inf", font_size=10, color=RED).next_to(mask_grid, RIGHT, buff=0.1)

        self.play(FadeIn(mask_grid), FadeIn(mask_label), FadeIn(neg_inf))
        self.wait(0.5)

        # Online Softmax
        softmax_box = RoundedRectangle(
            width=6, height=1.6, corner_radius=0.1,
            color=C_ATTN, fill_opacity=0.08, stroke_width=1.5,
        ).move_to(DOWN * 0.5)
        softmax_title = Text("Online Softmax (Flash Attention)", font_size=16, color=C_ATTN)
        softmax_title.next_to(softmax_box, UP, buff=0.05)

        step_texts = VGroup(
            MathTex(r"m_i = \max(m_i, \max(\text{qk}))", font_size=22, color=GREY_B),
            MathTex(r"\alpha = e^{m_{\text{old}} - m_{\text{new}}}", font_size=22, color=GREY_B),
            MathTex(r"\text{acc} = \alpha \cdot \text{acc} + P \cdot V", font_size=22, color=GREY_B),
            MathTex(r"\text{out} = \text{acc} / l_i", font_size=22, color=C_OUTPUT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        step_texts.move_to(softmax_box)

        self.play(FadeIn(softmax_box), FadeIn(softmax_title))
        self.play(LaggedStart(*[FadeIn(s) for s in step_texts], lag_ratio=0.25), run_time=2)

        # Memory comparison
        mem_text = VGroup(
            Text("Standard Attention: O(N^2) memory", font_size=16, color=RED_C),
            Text("Flash Attention:    O(N)  memory", font_size=16, color=C_OUTPUT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        mem_text.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(mem_text), run_time=1)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ====================================================================
# Scene 7: Transformer Block — Full Data Flow
# ====================================================================
class S07_TransformerBlock(Scene):
    def construct(self):
        title = Text("Transformer Block: Complete Data Flow", font_size=34, color=C_TILE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # Build the block diagram vertically
        x_in = make_tensor_block(2.5, 0.55, C_INPUT, "x", "(2, 64, 768)", fill_opacity=0.2)
        ln1 = make_tensor_block(2.5, 0.55, C_ACCENT, "LayerNorm 1", "", fill_opacity=0.15)
        attn = make_tensor_block(2.5, 0.55, C_ATTN, "Multi-Head Attention", "12 heads", fill_opacity=0.15)
        add1_label = Text("+ Residual", font_size=14, color=C_RESIDUAL)
        ln2 = make_tensor_block(2.5, 0.55, C_ACCENT, "LayerNorm 2", "", fill_opacity=0.15)
        mlp_expand = make_tensor_block(2.5, 0.55, C_WEIGHT, "Linear 768 -> 3072", "", fill_opacity=0.15)
        gelu = make_tensor_block(2.5, 0.55, C_GELU, "GELU", "", fill_opacity=0.15)
        mlp_contract = make_tensor_block(2.5, 0.55, C_WEIGHT, "Linear 3072 -> 768", "", fill_opacity=0.15)
        add2_label = Text("+ Residual", font_size=14, color=C_RESIDUAL)
        x_out = make_tensor_block(2.5, 0.55, C_OUTPUT, "x'", "(2, 64, 768)", fill_opacity=0.2)

        flow = VGroup(x_in, ln1, attn, add1_label, ln2, mlp_expand, gelu, mlp_contract, add2_label, x_out)
        flow.arrange(DOWN, buff=0.12)
        flow.scale(0.8).next_to(title, DOWN, buff=0.3)

        # Animate block by block
        for i, item in enumerate(flow):
            self.play(FadeIn(item, shift=DOWN * 0.15), run_time=0.3)
            if i < len(flow) - 1:
                a = data_flow_arrow(item.get_bottom(), flow[i + 1].get_top())
                self.play(GrowArrow(a), run_time=0.15)

        # Residual connection arcs
        res1_arc = CurvedArrow(
            x_in.get_left() + LEFT * 0.1,
            add1_label.get_left() + LEFT * 0.1,
            angle=-TAU / 4,
            color=C_RESIDUAL, stroke_width=2,
        )
        res2_arc = CurvedArrow(
            add1_label.get_right() + RIGHT * 0.3,
            add2_label.get_right() + RIGHT * 0.3,
            angle=TAU / 4,
            color=C_RESIDUAL, stroke_width=2,
        )
        self.play(Create(res1_arc), Create(res2_arc), run_time=1)

        # Shape annotations on the right
        shapes = VGroup(
            Text("(2,64,768)", font_size=11, color=GREY),
            Text("(2,64,768)", font_size=11, color=GREY),
            Text("(2,64,768)", font_size=11, color=GREY),
            Text("(2,64,768)", font_size=11, color=GREY),
            Text("(2,64,768)", font_size=11, color=GREY),
            Text("(2,64,3072)", font_size=11, color=GREY),
            Text("(2,64,3072)", font_size=11, color=GREY),
            Text("(2,64,768)", font_size=11, color=GREY),
            Text("(2,64,768)", font_size=11, color=GREY),
            Text("(2,64,768)", font_size=11, color=GREY),
        )
        for sh, item in zip(shapes, flow):
            sh.next_to(item, RIGHT, buff=0.15)
        self.play(FadeIn(shapes), run_time=0.8)

        repeat = Text("x 12 layers", font_size=20, color=C_TILE).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(repeat))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ====================================================================
# Scene 8: MHA Detail — QKV Projection + Head Split
# ====================================================================
class S08_MHADetail(Scene):
    def construct(self):
        title = Text("Multi-Head Attention: QKV Flow", font_size=34, color=C_ATTN)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Input
        x_block = make_tensor_block(2.0, 0.7, C_INPUT, "x_norm", "(2,64,768)")
        x_block.move_to(LEFT * 5 + UP * 1)

        # QKV projection
        qkv_block = make_tensor_block(2.2, 0.7, C_WEIGHT, "Linear", "768 -> 2304")
        qkv_block.move_to(LEFT * 2 + UP * 1)

        a1 = data_flow_arrow(x_block.get_right(), qkv_block.get_left())
        self.play(FadeIn(x_block))
        self.play(GrowArrow(a1), FadeIn(qkv_block))

        # Split into Q, K, V
        qkv_out = make_tensor_block(2.0, 0.7, C_TILE, "QKV", "(2,64,2304)")
        qkv_out.move_to(RIGHT * 1 + UP * 1)
        a2 = data_flow_arrow(qkv_block.get_right(), qkv_out.get_left())
        self.play(GrowArrow(a2), FadeIn(qkv_out))

        # Split into 3
        q = make_tensor_block(1.5, 0.55, C_INPUT, "Q", "(2,64,768)", fill_opacity=0.2)
        k = make_tensor_block(1.5, 0.55, C_WEIGHT, "K", "(2,64,768)", fill_opacity=0.2)
        v = make_tensor_block(1.5, 0.55, C_OUTPUT, "V", "(2,64,768)", fill_opacity=0.2)
        qkv_split = VGroup(q, k, v).arrange(DOWN, buff=0.15)
        qkv_split.move_to(RIGHT * 4 + UP * 1)

        split_arrows = VGroup()
        for block in [q, k, v]:
            sa = data_flow_arrow(qkv_out.get_right(), block.get_left())
            split_arrows.add(sa)
        self.play(LaggedStart(*[GrowArrow(a) for a in split_arrows], lag_ratio=0.1), FadeIn(qkv_split), run_time=1)

        # Head reshape
        reshape_text = Text("Reshape: (2, 64, 768) -> (2, 12, 64, 64)", font_size=14, color=GREY_B)
        reshape_text.move_to(DOWN * 0.5)
        self.play(FadeIn(reshape_text))

        # Show head split visually
        heads = VGroup()
        head_colors = [BLUE, GREEN, ORANGE, PURPLE, TEAL, RED, YELLOW, PINK, BLUE_B, GREEN_B, ORANGE, PURPLE_B]
        for i in range(12):
            h = Square(side_length=0.3, color=head_colors[i % len(head_colors)], fill_opacity=0.4, stroke_width=1)
            heads.add(h)
        heads.arrange_in_grid(rows=2, cols=6, buff=0.08)
        heads.move_to(DOWN * 1.5)
        heads_label = Text("12 Attention Heads — each (64, 64)", font_size=14, color=GREY_B)
        heads_label.next_to(heads, DOWN, buff=0.15)

        self.play(
            LaggedStart(*[FadeIn(h, scale=0.5) for h in heads], lag_ratio=0.05),
            FadeIn(heads_label),
            run_time=1.5,
        )

        parallel_text = Text("All heads computed in parallel on GPU", font_size=16, color=C_ACCENT)
        parallel_text.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(parallel_text))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ====================================================================
# Scene 9: Generation Loop — Autoregressive Token Production
# ====================================================================
class S09_Generation(Scene):
    def construct(self):
        title = Text("Autoregressive Generation", font_size=36, color=C_OUTPUT)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Sequence of tokens growing
        tokens = ["The", "future", "of", "GPU", "is"]
        generated = ["declarative", "programming", "."]
        all_words = tokens + generated

        token_boxes = VGroup()
        for i, word in enumerate(all_words):
            col = C_INPUT if i < len(tokens) else C_OUTPUT
            box = VGroup(
                RoundedRectangle(width=1.3, height=0.5, corner_radius=0.08, color=col, fill_opacity=0.3, stroke_width=1.5),
                Text(word, font_size=16, color=col),
            )
            box[1].move_to(box[0])
            token_boxes.add(box)

        token_boxes.arrange(RIGHT, buff=0.1)
        token_boxes.move_to(UP * 0.5)

        # Show input tokens first
        for i in range(len(tokens)):
            self.play(FadeIn(token_boxes[i], shift=RIGHT * 0.2), run_time=0.3)

        # Generate each new token with the model pipeline mini-diagram
        pipeline = VGroup(
            Text("Forward Pass", font_size=14, color=GREY_B),
            Text("->", font_size=14, color=GREY),
            Text("Logits (50257)", font_size=14, color=C_ACCENT),
            Text("->", font_size=14, color=GREY),
            Text("Softmax", font_size=14, color=C_ACCENT),
            Text("->", font_size=14, color=GREY),
            Text("Sample", font_size=14, color=C_OUTPUT),
        ).arrange(RIGHT, buff=0.1)
        pipeline.move_to(DOWN * 1)

        self.play(FadeIn(pipeline))

        for i in range(len(generated)):
            idx = len(tokens) + i
            # Flash the pipeline
            self.play(
                pipeline[0].animate.set_color(WHITE),
                run_time=0.2,
            )
            self.play(
                pipeline[2].animate.set_color(WHITE),
                run_time=0.2,
            )
            self.play(
                pipeline[4].animate.set_color(WHITE),
                run_time=0.2,
            )
            self.play(
                pipeline[6].animate.set_color(WHITE),
                run_time=0.2,
            )
            # New token appears
            self.play(FadeIn(token_boxes[idx], shift=UP * 0.3), run_time=0.4)
            # Reset pipeline colors
            self.play(
                pipeline[0].animate.set_color(GREY_B),
                pipeline[2].animate.set_color(C_ACCENT),
                pipeline[4].animate.set_color(C_ACCENT),
                pipeline[6].animate.set_color(C_OUTPUT),
                run_time=0.15,
            )

        # Highlight the growing sequence
        seq_rect = SurroundingRectangle(token_boxes, color=C_OUTPUT, buff=0.1, stroke_width=2)
        self.play(Create(seq_rect))

        context_note = Text(
            "Each step: full forward pass through 12 transformer blocks",
            font_size=15, color=GREY_B,
        ).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(context_note))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ====================================================================
# Scene 10: Tile vs CUDA Comparison — Visual
# ====================================================================
class S10_TileVsCuda(Scene):
    def construct(self):
        title = Text("Tile Philosophy vs Traditional CUDA", font_size=34, color=C_TILE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Left side: CUDA threads chaos
        cuda_label = Text("CUDA: Manage Threads", font_size=18, color=RED_C)
        cuda_label.move_to(LEFT * 3.5 + UP * 1.5)

        # Dots representing individual threads (chaotic)
        cuda_threads = VGroup()
        np.random.seed(42)
        for _ in range(40):
            d = Dot(
                point=LEFT * 3.5 + np.random.uniform(-1.2, 1.2) * RIGHT + np.random.uniform(-0.8, 0.8) * UP + DOWN * 0.3,
                radius=0.04, color=RED,
            )
            cuda_threads.add(d)

        # Sync barriers
        sync_lines = VGroup()
        for y in [-0.2, 0.5]:
            line = DashedLine(
                LEFT * 4.7 + UP * y, LEFT * 2.3 + UP * y,
                color=RED, dash_length=0.1,
            )
            sync_lines.add(line)
        sync_label = Text("__syncthreads()", font_size=10, color=RED).next_to(sync_lines[0], RIGHT, buff=0.05)

        self.play(FadeIn(cuda_label))
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.3) for d in cuda_threads], lag_ratio=0.02),
            run_time=1,
        )
        self.play(Create(sync_lines), FadeIn(sync_label))

        # Right side: Tile blocks (organized)
        tile_label = Text("Tile: Declare Data Ops", font_size=18, color=C_TILE)
        tile_label.move_to(RIGHT * 3.5 + UP * 1.5)

        tile_blocks = VGroup()
        tile_ops = ["load", "mma", "store", "load", "gelu", "store"]
        for i, op in enumerate(tile_ops):
            r = i // 3
            c = i % 3
            box = VGroup(
                RoundedRectangle(
                    width=0.9, height=0.5, corner_radius=0.06,
                    color=C_TILE, fill_opacity=0.3, stroke_width=1.5,
                ),
                Text(op, font_size=12, color=C_TILE),
            )
            box[1].move_to(box[0])
            box.move_to(RIGHT * (2.5 + c * 1.0) + DOWN * (0.3 - r * 0.7))
            tile_blocks.add(box)

        self.play(FadeIn(tile_label))
        self.play(
            LaggedStart(*[FadeIn(b, shift=DOWN * 0.2) for b in tile_blocks], lag_ratio=0.1),
            run_time=1.5,
        )

        # Compiler arrow
        compiler_arrow = Arrow(
            RIGHT * 3.5 + DOWN * 1.3,
            RIGHT * 3.5 + DOWN * 2.0,
            color=C_ACCENT, stroke_width=2,
        )
        compiler_label = Text("Compiler optimizes automatically", font_size=13, color=C_ACCENT)
        compiler_label.next_to(compiler_arrow, RIGHT, buff=0.1)
        self.play(GrowArrow(compiler_arrow), FadeIn(compiler_label))

        # Divider
        divider = Line(UP * 1.8, DOWN * 2.5, color=GREY, stroke_width=1).move_to(ORIGIN)
        self.play(Create(divider))

        # Bottom comparison
        comparison = VGroup(
            Text("threadIdx, __syncthreads, __shared__", font_size=13, color=RED_C),
            Text("ct.load, ct.mma, ct.store", font_size=13, color=C_TILE),
        ).arrange(RIGHT, buff=2)
        comparison.to_edge(DOWN, buff=0.5)

        vs = Text("vs", font_size=16, color=GREY).move_to(comparison.get_center())
        self.play(FadeIn(comparison), FadeIn(vs))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ====================================================================
# Scene 11: Closing — Full Pipeline Summary
# ====================================================================
class S11_Closing(Scene):
    def construct(self):
        # Full pipeline in one view
        pipeline_title = Text("cutileGPT: End-to-End", font_size=36, color=C_INPUT)
        pipeline_title.to_edge(UP, buff=0.3)
        self.play(Write(pipeline_title))

        # Compact pipeline
        stages = [
            ("Tokens", C_INPUT, "(B, T)"),
            ("Embed", C_TILE, "+pos"),
            ("LN", C_ACCENT, "norm"),
            ("Attn", C_ATTN, "flash"),
            ("+res", C_RESIDUAL, ""),
            ("LN", C_ACCENT, "norm"),
            ("MLP", C_WEIGHT, "768->3072->768"),
            ("GELU", C_GELU, ""),
            ("+res", C_RESIDUAL, ""),
            ("LN_f", C_ACCENT, "final"),
            ("Logits", C_OUTPUT, "(B,T,V)"),
        ]

        blocks = VGroup()
        for name, col, sub in stages:
            box = VGroup(
                RoundedRectangle(
                    width=1.0, height=0.7, corner_radius=0.06,
                    color=col, fill_opacity=0.25, stroke_width=1.5,
                ),
                Text(name, font_size=13, color=col, weight=BOLD),
            )
            box[1].move_to(box[0].get_center() + UP * 0.05)
            if sub:
                s = Text(sub, font_size=9, color=GREY_B)
                s.move_to(box[0].get_center() + DOWN * 0.15)
                box.add(s)
            blocks.add(box)

        # Two rows
        row1 = VGroup(*blocks[:6]).arrange(RIGHT, buff=0.15)
        row2 = VGroup(*blocks[6:]).arrange(RIGHT, buff=0.15)
        all_rows = VGroup(row1, row2).arrange(DOWN, buff=0.4)
        all_rows.move_to(UP * 0.3)

        self.play(
            LaggedStart(*[FadeIn(b, shift=RIGHT * 0.2) for b in blocks], lag_ratio=0.08),
            run_time=2.5,
        )

        # Repeat bracket for 12 layers
        bracket_start = blocks[2].get_top() + UP * 0.15
        bracket_end = blocks[8].get_top() + UP * 0.15
        brace = Brace(VGroup(blocks[2], blocks[3], blocks[4], blocks[5], blocks[6], blocks[7], blocks[8]), UP, buff=0.1, color=C_TILE)
        brace_label = Text("x 12 layers", font_size=14, color=C_TILE).next_to(brace, UP, buff=0.05)
        self.play(GrowFromCenter(brace), FadeIn(brace_label), run_time=1)

        # Key stats at the bottom
        stats = VGroup(
            Text("8.3x faster kernels", font_size=20, color=GREEN),
            Text("87% less code", font_size=20, color=C_ACCENT),
            Text("Zero thread management", font_size=20, color=C_TILE),
        ).arrange(RIGHT, buff=1)
        stats.to_edge(DOWN, buff=0.5)

        self.play(
            LaggedStart(*[FadeIn(s, shift=UP * 0.2) for s in stats], lag_ratio=0.2),
            run_time=1.5,
        )

        self.wait(2)

        # Final title
        self.play(*[FadeOut(m) for m in self.mobjects])
        final = Text("cutileGPT", font_size=72, color=C_INPUT)
        final_sub = Text("Think in WHAT, not HOW", font_size=28, color=GREY_B).next_to(final, DOWN, buff=0.4)
        self.play(Write(final), run_time=1)
        self.play(FadeIn(final_sub))
        self.wait(2)
        self.play(FadeOut(final), FadeOut(final_sub))
