# SPDX-License-Identifier: Apache-2.0
"""Static-shape CUDA Graph execution for cutileGPT inference."""

from __future__ import annotations

from typing import Any, Mapping

import cupy as cp


class CUDAGraphForward:
    """Capture and replay a fixed-shape, cache-free model forward pass.

    CUDA Graph nodes retain the output/intermediate addresses selected during
    capture. The returned output is therefore a stable buffer overwritten by
    every replay; copy it if it must outlive the next call.

    Autoregressive cache advancement is intentionally excluded for now: cache
    length changes kernel arguments and array shapes. Decode graphs need a
    bucketed fixed-capacity cache with a device-side position scalar rather
    than silently capturing one length forever.
    """

    def __init__(
        self,
        model: Any,
        example_idx: cp.ndarray,
        *,
        warmup: int = 3,
        forward_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if warmup < 1:
            raise ValueError("warmup must be at least 1")
        if not isinstance(example_idx, cp.ndarray):
            raise ValueError("example_idx must be a CuPy array")

        self.model = model
        self.forward_kwargs = dict(forward_kwargs or {})
        if self.forward_kwargs.get("cache") is not None:
            raise ValueError("cache-mutating forwards cannot be captured safely")
        self.input = cp.empty_like(example_idx)
        cp.copyto(self.input, example_idx)
        capture_stream = cp.cuda.Stream(non_blocking=True)

        with capture_stream:
            for _ in range(warmup):
                model.forward(self.input, **self.forward_kwargs)
        capture_stream.synchronize()

        with capture_stream:
            capture_stream.begin_capture()
            result = model.forward(self.input, **self.forward_kwargs)
            self.graph = capture_stream.end_capture()

        # A graph retains device pointers, not the Python objects owning them.
        # Models with replaceable external buffers (for example a growing RoPE
        # table) expose those buffers here so a later model call cannot return
        # captured storage to the memory pool.
        resource_hook = getattr(model, "graph_capture_resources", None)
        self._retained_resources = (
            tuple(resource_hook()) if resource_hook is not None else ()
        )
        self._tuple_result = isinstance(result, tuple)
        self.output = result[0] if self._tuple_result else result
        self.graph.upload(cp.cuda.get_current_stream())

    def replay(self, idx: cp.ndarray, *, synchronize: bool = False):
        """Copy new token IDs into the static input and enqueue one graph replay."""
        if idx.shape != self.input.shape or idx.dtype != self.input.dtype:
            raise ValueError(
                f"expected input {self.input.shape} {self.input.dtype}, "
                f"got {idx.shape} {idx.dtype}"
            )
        stream = cp.cuda.get_current_stream()
        cp.copyto(self.input, idx)
        self.graph.launch(stream)
        if synchronize:
            stream.synchronize()
        if self._tuple_result:
            return self.output, None
        return self.output


def capture_forward(
    model: Any,
    example_idx: cp.ndarray,
    *,
    warmup: int = 3,
    forward_kwargs: Mapping[str, Any] | None = None,
) -> CUDAGraphForward:
    """Convenience constructor for :class:`CUDAGraphForward`."""
    return CUDAGraphForward(
        model,
        example_idx,
        warmup=warmup,
        forward_kwargs=forward_kwargs,
    )
