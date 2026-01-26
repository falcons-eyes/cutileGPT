# SPDX-License-Identifier: Apache-2.0
"""
GELU activation function kernel for cutile GPT.

Implements the approximate GELU used in GPT-2:
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
"""

import math
import cupy as cp
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Constants for GELU approximation
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)  # ~0.7978845608
GELU_COEF = 0.044715


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def gelu_kernel(X, Y, TILE_SIZE: ConstInt):
    """
    Element-wise GELU activation kernel.

    Args:
        X: Input tensor (flattened to 1D for processing)
        Y: Output tensor (same shape as X)
        TILE_SIZE: Number of elements processed per block
    """
    bid = ct.bid(0)

    # Load input tile
    x = ct.load(X, index=(bid,), shape=(TILE_SIZE,), padding_mode=ct.PaddingMode.ZERO)

    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_cubed = x * x * x
    inner = SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed)
    y = 0.5 * x * (1.0 + ct.tanh(inner))

    # Store result
    ct.store(Y, index=(bid,), tile=y.astype(Y.dtype))


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1), occupancy=4)
def gelu_kernel_2d(X, Y, TILE_M: ConstInt, TILE_N: ConstInt):
    """
    2D GELU activation kernel for matrices.

    Args:
        X: Input tensor (M, N)
        Y: Output tensor (M, N)
        TILE_M: Tile size along M dimension
        TILE_N: Tile size along N dimension
    """
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # Load input tile
    x = ct.load(X, index=(bid_m, bid_n), shape=(TILE_M, TILE_N),
                padding_mode=ct.PaddingMode.ZERO)

    # GELU approximation
    x_cubed = x * x * x
    inner = SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed)
    y = 0.5 * x * (1.0 + ct.tanh(inner))

    # Store result
    ct.store(Y, index=(bid_m, bid_n), tile=y.astype(Y.dtype))


def cutile_gelu(x: cp.ndarray) -> cp.ndarray:
    """
    Apply GELU activation using cutile kernel.

    Args:
        x: Input tensor of any shape

    Returns:
        Output tensor with GELU applied element-wise
    """
    if not isinstance(x, cp.ndarray):
        raise ValueError("Input tensor must be a CuPy array on CUDA device")

    # For 2D tensors, use 2D kernel
    if x.ndim == 2:
        M, N = x.shape
        y = cp.empty_like(x)

        TILE_M, TILE_N = 32, 128
        grid_m = math.ceil(M / TILE_M)
        grid_n = math.ceil(N / TILE_N)
        grid = (grid_m, grid_n, 1)

        ct.launch(cp.cuda.get_current_stream(), grid, gelu_kernel_2d,
                  (x, y, TILE_M, TILE_N))
        return y

    # For other shapes, flatten and use 1D kernel
    original_shape = x.shape
    x_flat = cp.reshape(x, -1)
    y_flat = cp.empty_like(x_flat)

    TILE_SIZE = 1024
    num_elements = x_flat.size
    grid = (math.ceil(num_elements / TILE_SIZE), 1, 1)

    ct.launch(cp.cuda.get_current_stream(), grid, gelu_kernel,
              (x_flat, y_flat, TILE_SIZE))

    return cp.reshape(y_flat, original_shape)


# Reference CuPy implementation for testing
def cupy_gelu(x: cp.ndarray) -> cp.ndarray:
    """CuPy reference GELU (approximate version matching GPT-2)"""
    return 0.5 * x * (1.0 + cp.tanh(SQRT_2_OVER_PI * (x + GELU_COEF * cp.power(x, 3.0))))


if __name__ == "__main__":
    print("--- Testing cutile GELU kernel ---")

    # Test 1D
    x = cp.random.randn(1024, dtype=cp.float32)
    y_cutile = cutile_gelu(x)
    y_cupy = cupy_gelu(x)

    print(f"1D Test - Max diff: {cp.abs(y_cutile - y_cupy).max():.6f}")
    cp.testing.assert_allclose(y_cutile, y_cupy, atol=1e-5, rtol=1e-5)
    print("1D Test passed!")

    # Test 2D
    x = cp.random.randn(128, 256, dtype=cp.float32)
    y_cutile = cutile_gelu(x)
    y_cupy = cupy_gelu(x)

    print(f"2D Test - Max diff: {cp.abs(y_cutile - y_cupy).max():.6f}")
    cp.testing.assert_allclose(y_cutile, y_cupy, atol=1e-5, rtol=1e-5)
    print("2D Test passed!")

    # Test 3D (typical transformer shape: batch, seq, hidden)
    x = cp.random.randn(2, 64, 48, dtype=cp.float32)
    y_cutile = cutile_gelu(x)
    y_cupy = cupy_gelu(x)

    print(f"3D Test - Max diff: {cp.abs(y_cutile - y_cupy).max():.6f}")
    cp.testing.assert_allclose(y_cutile, y_cupy, atol=1e-5, rtol=1e-5)
    print("3D Test passed!")

    print("\n--- All GELU tests passed! ---")
