# SPDX-License-Identifier: Apache-2.0
"""KV caching and sliding-window attention.

Without a cache, generating N tokens re-runs attention over the whole prefix
every step, so the cost grows with the square of the sequence. These check that
caching changes the cost and not the answer, and that a windowed layer matches
an explicitly masked reference.
"""
import pytest

cp = pytest.importorskip("cupy")
torch = pytest.importorskip("torch")

from cutile_gpt import CutileGPT, GPTConfig, KVCache, cutile_causal_attention


def gpu_available():
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not gpu_available(), reason="needs a CUDA GPU")


def to_cupy(t):
    return cp.from_dlpack(t.contiguous())


def test_cached_decode_matches_full_attention():
    """One query row against a cached history must equal the corresponding row
    of attention computed over the whole sequence at once."""
    batch, n_head, n_kv_head, head_dim, total = 1, 8, 2, 64, 128
    torch.manual_seed(0)
    q = torch.randn(batch, n_head, total, head_dim, device="cuda")
    k = torch.randn(batch, n_kv_head, total, head_dim, device="cuda")
    v = torch.randn(batch, n_kv_head, total, head_dim, device="cuda")

    full = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=True)

    cache = KVCache(1, batch, n_kv_head, total, head_dim)
    for step in range(total):
        ck, cv = cache.append(
            0, to_cupy(k[:, :, step:step + 1]), to_cupy(v[:, :, step:step + 1]))
        out = cutile_causal_attention(
            to_cupy(q[:, :, step:step + 1]), ck, cv, n_head, n_kv_head)
        cp.cuda.Stream.null.synchronize()
        err = (torch.from_dlpack(out)[:, :, 0] - full[:, :, step]).abs().max()
        assert err.item() < 1e-5, f"step {step}: {err.item()}"

    assert cache.length == total


def test_cache_holds_kv_heads_not_query_heads():
    """The GQA sharing that makes the kernel cheap shrinks the cache too."""
    mha = KVCache(32, 1, 32, 4096, 128)
    gqa = KVCache(32, 1, 2, 4096, 128)
    assert mha.nbytes == gqa.nbytes * 16


def test_cache_rejects_overflow():
    cache = KVCache(1, 1, 2, 8, 64)
    kv = to_cupy(torch.randn(1, 2, 9, 64, device="cuda"))
    with pytest.raises(ValueError, match="overflow"):
        cache.append(0, kv, kv)


def test_cache_rejects_wrong_shape():
    cache = KVCache(1, 1, 2, 16, 64)
    kv = to_cupy(torch.randn(1, 4, 2, 64, device="cuda"))
    with pytest.raises(ValueError, match="n_kv_head"):
        cache.append(0, kv, kv)


def test_cache_reset_keeps_buffers():
    cache = KVCache(1, 1, 2, 16, 64)
    kv = to_cupy(torch.randn(1, 2, 4, 64, device="cuda"))
    cache.append(0, kv, kv)
    assert cache.length == 4
    allocated = cache.nbytes
    cache.reset()
    assert cache.length == 0
    assert cache.nbytes == allocated


@pytest.mark.parametrize("window", [32, 64, 128, 256])
def test_sliding_window_matches_masked_reference(window):
    """A row can have a whole KV tile fall outside its window, which leaves the
    running softmax max at -inf; the kernel has to survive that."""
    batch, n_head, n_kv_head, head_dim, seq = 1, 8, 2, 64, 512
    torch.manual_seed(0)
    q = torch.randn(batch, n_head, seq, head_dim, device="cuda")
    k = torch.randn(batch, n_kv_head, seq, head_dim, device="cuda")
    v = torch.randn(batch, n_kv_head, seq, head_dim, device="cuda")

    got = cutile_causal_attention(
        to_cupy(q), to_cupy(k), to_cupy(v), n_head, n_kv_head, window=window)
    cp.cuda.Stream.null.synchronize()

    idx = torch.arange(seq, device="cuda")
    allowed = (idx[:, None] >= idx[None, :]) & (idx[None, :] > idx[:, None] - window)
    bias = torch.zeros(seq, seq, device="cuda").masked_fill(~allowed, float("-inf"))
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, enable_gqa=True)

    assert (torch.from_dlpack(got) - expected).abs().max().item() < 1e-4


def test_window_zero_is_full_causal():
    batch, n_head, head_dim, seq = 1, 4, 64, 256
    torch.manual_seed(0)
    q, k, v = (torch.randn(batch, n_head, seq, head_dim, device="cuda")
               for _ in range(3))
    windowed = cutile_causal_attention(
        to_cupy(q), to_cupy(k), to_cupy(v), n_head, window=0)
    plain = cutile_causal_attention(to_cupy(q), to_cupy(k), to_cupy(v), n_head)
    cp.cuda.Stream.null.synchronize()
    assert float(cp.abs(windowed - plain).max()) == 0.0


def test_generate_agrees_with_and_without_cache():
    """The cache must be a speed change, not a behaviour change."""
    model = CutileGPT(GPTConfig.gpt_nano())
    start = cp.array([[100, 200, 300, 400]], dtype=cp.int32)

    for n in (1, 8, 32):
        cp.random.seed(0)
        cached = model.generate(start, max_new_tokens=n, use_cache=True)
        cp.random.seed(0)
        uncached = model.generate(start, max_new_tokens=n, use_cache=False)
        assert bool((cached == uncached).all()), f"diverged at {n} tokens"
