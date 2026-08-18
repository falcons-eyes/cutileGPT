# SPDX-License-Identifier: Apache-2.0
"""Every current open-weight model ships bfloat16, so the whole path to it -
dtype registration, the DLPack hand-off, and the kernels themselves - has to
keep working. Reaching bfloat16 needs ml_dtypes for the numpy dtype and cupy 14
for `from_dlpack`; before both floors were declared, loading any non-fp32
checkpoint died on "Got unsupported ScalarType BFloat16".

Kernel cases need a Blackwell GPU and are skipped without one. The dtype and
loader cases run anywhere cupy imports.
"""
import pytest

cp = pytest.importorskip("cupy")
torch = pytest.importorskip("torch")

from cutile_gpt.api.config import DType
from cutile_gpt.models.gpt import _torch_to_cupy

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# bfloat16 has an 8-bit mantissa, so ~1e-2 absolute error on unit-scale
# activations is the format working correctly, not a kernel bug.
TOLERANCE = {torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 5e-2}


def gpu_available():
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not gpu_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("dtype", list(DType))
def test_to_cupy_resolves_every_dtype(dtype):
    """A dict of all three would evaluate bfloat16 eagerly and take fp32 down
    with it - the bug this guards against broke every member, not just one."""
    assert dtype.to_cupy() is not None


def test_bfloat16_dtype_is_registered():
    assert cp.dtype("bfloat16").itemsize == 2


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_loader_preserves_dtype_losslessly(dtype):
    """The old loader went through numpy, which has no bfloat16."""
    if dtype is not torch.float32 and not gpu_available():
        pytest.skip("needs a CUDA GPU")
    tensor = torch.randn(64, 128, dtype=dtype)
    array = _torch_to_cupy(tensor)

    assert array.dtype.itemsize == tensor.element_size()
    roundtrip = torch.from_dlpack(array).cpu().float()
    assert (roundtrip - tensor.float()).abs().max().item() == 0.0


@requires_gpu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_gelu_kernel_accepts_dtype(dtype):
    from cutile_gpt.kernels.gelu import cutile_gelu

    x = cp.from_dlpack(torch.randn(4, 128, 768, dtype=dtype, device="cuda"))
    out = cutile_gelu(x)
    cp.cuda.Stream.null.synchronize()

    assert out.dtype == x.dtype
    expected = torch.nn.functional.gelu(
        torch.from_dlpack(x).float(), approximate="tanh"
    )
    err = (torch.from_dlpack(out).float() - expected).abs().max().item()
    assert err < TOLERANCE[dtype], f"{dtype}: {err}"


@requires_gpu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_attention_kernel_accepts_dtype(dtype):
    from cutile_gpt.kernels.attention import cutile_causal_attention

    shape = (1, 8, 256, 64)
    q, k, v = (
        cp.from_dlpack(torch.randn(*shape, dtype=dtype, device="cuda"))
        for _ in range(3)
    )
    out = cutile_causal_attention(q, k, v, shape[1])
    cp.cuda.Stream.null.synchronize()

    assert out.dtype == q.dtype
    expected = torch.nn.functional.scaled_dot_product_attention(
        *(torch.from_dlpack(t).float() for t in (q, k, v)), is_causal=True
    )
    err = (torch.from_dlpack(out).float() - expected).abs().max().item()
    assert err < TOLERANCE[dtype], f"{dtype}: {err}"
