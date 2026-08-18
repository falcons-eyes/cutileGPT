# SPDX-License-Identifier: Apache-2.0
"""The primitives current open-weight models are built from.

GPT-2 used LayerNorm, MHA, and a GELU MLP. Qwen3, Gemma, Llama, and Muse
Glimmer all use RMSNorm, grouped-query attention, and a gated SwiGLU MLP
instead. These check each against PyTorch, then compose them into a decoder
layer shaped like Muse Glimmer's.

Everything here needs a Blackwell GPU and is skipped without one.
"""
import pytest

cp = pytest.importorskip("cupy")
torch = pytest.importorskip("torch")

from cutile_gpt import (
    cutile_causal_attention,
    cutile_linear,
    cutile_rms_norm,
    cutile_rope,
    cutile_swiglu_mlp,
    rope_tables,
)


def gpu_available():
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not gpu_available(), reason="needs a CUDA GPU")

# Projections run on TF32 tensor cores, which carry ~2e-3 absolute error at
# these magnitudes. cutile_linear matches torch's own TF32 matmul exactly, so
# anything at this scale is the format, not the kernel.
TF32_TOL = 5e-3


def to_cupy(t):
    return cp.from_dlpack(t.contiguous())


@pytest.mark.parametrize("n_embd", [48, 768, 6656])
def test_rms_norm_matches_torch(n_embd):
    torch.manual_seed(0)
    x = torch.randn(2, 64, n_embd, device="cuda")
    w = torch.randn(n_embd, device="cuda")

    got = cutile_rms_norm(to_cupy(x), to_cupy(w), eps=1e-6)
    cp.cuda.Stream.null.synchronize()
    expected = torch.nn.functional.rms_norm(x, (n_embd,), w, eps=1e-6)

    assert (torch.from_dlpack(got) - expected).abs().max().item() < 1e-5


def test_rms_norm_unit_offset_is_gemma_convention():
    """Gemma stores RMSNorm weights centered on zero and scales by (1 + w)."""
    torch.manual_seed(0)
    n_embd = 768
    x = torch.randn(2, 64, n_embd, device="cuda")
    w = torch.randn(n_embd, device="cuda")

    got = cutile_rms_norm(to_cupy(x), to_cupy(w), eps=1e-6, unit_offset=True)
    cp.cuda.Stream.null.synchronize()
    expected = torch.nn.functional.rms_norm(x, (n_embd,), w + 1.0, eps=1e-6)

    assert (torch.from_dlpack(got) - expected).abs().max().item() < 1e-5


@pytest.mark.parametrize(
    "n_head,n_kv_head",
    [(32, 32), (32, 8), (32, 2), (8, 1)],
    ids=["mha", "qwen3-style", "glimmer-style", "mqa"],
)
def test_grouped_query_attention_matches_torch(n_head, n_kv_head):
    """n_kv_head == n_head must still be plain MHA - the GQA index collapses
    back to head_idx, so this is also the backward-compatibility check."""
    torch.manual_seed(0)
    batch, seq, head_dim = 1, 256, 128
    q = torch.randn(batch, n_head, seq, head_dim, device="cuda")
    k = torch.randn(batch, n_kv_head, seq, head_dim, device="cuda")
    v = torch.randn(batch, n_kv_head, seq, head_dim, device="cuda")

    got = cutile_causal_attention(to_cupy(q), to_cupy(k), to_cupy(v), n_head)
    cp.cuda.Stream.null.synchronize()
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=(n_head != n_kv_head)
    )

    assert (torch.from_dlpack(got) - expected).abs().max().item() < 1e-4


def test_attention_rejects_indivisible_head_counts():
    q = to_cupy(torch.randn(1, 6, 64, 64, device="cuda"))
    kv = to_cupy(torch.randn(1, 4, 64, 64, device="cuda"))
    with pytest.raises(ValueError, match="divisible"):
        cutile_causal_attention(q, kv, kv, 6, n_kv_head=4)


def test_attention_rejects_mismatched_kv_shape():
    q = to_cupy(torch.randn(1, 8, 64, 64, device="cuda"))
    kv = to_cupy(torch.randn(1, 4, 64, 64, device="cuda"))
    with pytest.raises(ValueError, match="n_kv_head"):
        cutile_causal_attention(q, kv, kv, 8, n_kv_head=2)


@pytest.mark.parametrize(
    "n_embd,n_hidden", [(768, 3072), (6656, 19968)], ids=["gpt2-size", "glimmer"]
)
def test_swiglu_mlp_matches_torch(n_embd, n_hidden):
    torch.manual_seed(0)
    x = torch.randn(2, 64, n_embd, device="cuda") * 0.1
    gate = torch.randn(n_hidden, n_embd, device="cuda") * 0.02
    up = torch.randn(n_hidden, n_embd, device="cuda") * 0.02
    down = torch.randn(n_embd, n_hidden, device="cuda") * 0.02

    got = cutile_swiglu_mlp(*(to_cupy(t) for t in (x, gate, up, down)))
    cp.cuda.Stream.null.synchronize()
    expected = (
        torch.nn.functional.silu(x @ gate.T) * (x @ up.T)
    ) @ down.T

    assert (torch.from_dlpack(got) - expected).abs().max().item() < TF32_TOL


def test_swiglu_rejects_mismatched_projections():
    x = to_cupy(torch.randn(2, 8, 64, device="cuda"))
    gate = to_cupy(torch.randn(128, 64, device="cuda"))
    up = to_cupy(torch.randn(256, 64, device="cuda"))
    down = to_cupy(torch.randn(64, 128, device="cuda"))
    with pytest.raises(ValueError, match="gate and up"):
        cutile_swiglu_mlp(x, gate, up, down)


def _hf_rope(x, cos, sin):
    """HuggingFace takes full-width tables with each half duplicated."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    tc, ts = torch.from_dlpack(cos), torch.from_dlpack(sin)
    out, _ = apply_rotary_pos_emb(
        x, x, torch.cat([tc, tc], -1)[None], torch.cat([ts, ts], -1)[None]
    )
    return out


@pytest.mark.parametrize(
    "head_dim,theta",
    [(64, 10000.0), (128, 500000.0)],
    ids=["llama-theta", "glimmer-theta"],
)
def test_rope_matches_huggingface(head_dim, theta):
    torch.manual_seed(0)
    seq = 256
    x = torch.randn(1, 8, seq, head_dim, device="cuda")
    cos, sin = rope_tables(seq, head_dim, theta=theta)

    got = cutile_rope(to_cupy(x), cos, sin)
    cp.cuda.Stream.null.synchronize()

    assert (torch.from_dlpack(got) - _hf_rope(x, cos, sin)).abs().max().item() < 1e-5


def test_rope_offset_matches_absolute_positions():
    """Decoding with a cache rotates a new token by its absolute position, not
    by zero, so an offset run must equal the tail of a full-length run."""
    head_dim, theta = 128, 500000.0
    full_cos, full_sin = rope_tables(300, head_dim, theta=theta)
    off_cos, off_sin = rope_tables(44, head_dim, theta=theta, offset=256)

    assert float(cp.abs(full_cos[256:300] - off_cos).max()) == 0.0
    assert float(cp.abs(full_sin[256:300] - off_sin).max()) == 0.0


def test_rope_shared_between_query_and_key_head_counts():
    """Under GQA, Q and K have different head counts but one set of tables."""
    torch.manual_seed(0)
    head_dim, seq = 128, 256
    cos, sin = rope_tables(seq, head_dim, theta=500000.0)
    q = torch.randn(1, 32, seq, head_dim, device="cuda")
    k = torch.randn(1, 2, seq, head_dim, device="cuda")

    gq = cutile_rope(to_cupy(q), cos, sin)
    gk = cutile_rope(to_cupy(k), cos, sin)
    cp.cuda.Stream.null.synchronize()

    assert (torch.from_dlpack(gq) - _hf_rope(q, cos, sin)).abs().max().item() < 1e-5
    assert (torch.from_dlpack(gk) - _hf_rope(k, cos, sin)).abs().max().item() < 1e-5


def test_rope_rejects_odd_head_dim():
    with pytest.raises(ValueError, match="even"):
        rope_tables(16, 63)


def test_rope_rejects_mismatched_tables():
    x = to_cupy(torch.randn(1, 4, 128, 64, device="cuda"))
    cos, sin = rope_tables(64, 64)
    with pytest.raises(ValueError, match="cos/sin"):
        cutile_rope(x, cos, sin)


def test_muse_glimmer_decoder_layer():
    """A complete Glimmer decoder layer: RMSNorm, RoPE, GQA, SwiGLU.

    Note hidden (6656) is not n_head * head_dim (4096) - the output projection
    widens the attention result back up.
    """
    hidden, n_head, n_kv_head, head_dim, intermediate = 6656, 32, 2, 128, 19968
    attn_dim = n_head * head_dim
    batch, seq, eps, theta = 1, 128, 1e-6, 500000.0

    torch.manual_seed(0)
    w = {
        "ln_attn": torch.ones(hidden, device="cuda"),
        "ln_mlp": torch.ones(hidden, device="cuda"),
        "q": torch.randn(attn_dim, hidden, device="cuda") * 0.02,
        "k": torch.randn(n_kv_head * head_dim, hidden, device="cuda") * 0.02,
        "v": torch.randn(n_kv_head * head_dim, hidden, device="cuda") * 0.02,
        "o": torch.randn(hidden, attn_dim, device="cuda") * 0.02,
        "gate": torch.randn(intermediate, hidden, device="cuda") * 0.02,
        "up": torch.randn(intermediate, hidden, device="cuda") * 0.02,
        "down": torch.randn(hidden, intermediate, device="cuda") * 0.02,
    }
    x = torch.randn(batch, seq, hidden, device="cuda")

    cw = {name: to_cupy(t) for name, t in w.items()}
    cx = to_cupy(x)

    h = cutile_rms_norm(cx, cw["ln_attn"], eps=eps)
    h2d = cp.reshape(h, (-1, hidden))
    q = cp.ascontiguousarray(
        cp.reshape(cutile_linear(h2d, cw["q"]), (batch, seq, n_head, head_dim))
        .transpose(0, 2, 1, 3))
    k = cp.ascontiguousarray(
        cp.reshape(cutile_linear(h2d, cw["k"]), (batch, seq, n_kv_head, head_dim))
        .transpose(0, 2, 1, 3))
    v = cp.ascontiguousarray(
        cp.reshape(cutile_linear(h2d, cw["v"]), (batch, seq, n_kv_head, head_dim))
        .transpose(0, 2, 1, 3))
    cos, sin = rope_tables(seq, head_dim, theta=theta)
    q = cutile_rope(q, cos, sin)
    k = cutile_rope(k, cos, sin)
    attn = cutile_causal_attention(q, k, v, n_head, n_kv_head)
    attn = cp.ascontiguousarray(attn.transpose(0, 2, 1, 3)).reshape(-1, attn_dim)
    mid = cx + cp.reshape(cutile_linear(attn, cw["o"]), (batch, seq, hidden))
    h = cutile_rms_norm(mid, cw["ln_mlp"], eps=eps)
    got = mid + cutile_swiglu_mlp(h, cw["gate"], cw["up"], cw["down"])
    cp.cuda.Stream.null.synchronize()

    th = torch.nn.functional.rms_norm(x, (hidden,), w["ln_attn"], eps=eps)
    th2d = th.reshape(-1, hidden)
    tq = (th2d @ w["q"].T).view(batch, seq, n_head, head_dim).transpose(1, 2)
    tk = (th2d @ w["k"].T).view(batch, seq, n_kv_head, head_dim).transpose(1, 2)
    tv = (th2d @ w["v"].T).view(batch, seq, n_kv_head, head_dim).transpose(1, 2)
    tq = _hf_rope(tq, cos, sin)
    tk = _hf_rope(tk, cos, sin)
    tattn = torch.nn.functional.scaled_dot_product_attention(
        tq, tk, tv, is_causal=True, enable_gqa=True)
    tattn = tattn.transpose(1, 2).reshape(-1, attn_dim)
    tmid = x + (tattn @ w["o"].T).view(batch, seq, hidden)
    th = torch.nn.functional.rms_norm(tmid, (hidden,), w["ln_mlp"], eps=eps)
    expected = tmid + (
        torch.nn.functional.silu(th @ w["gate"].T) * (th @ w["up"].T)
    ) @ w["down"].T

    out = torch.from_dlpack(got)
    assert out.shape == expected.shape
    relative = ((out - expected).abs().max() / expected.abs().max()).item()
    assert relative < 5e-3, f"relative error {relative:.2e}"
