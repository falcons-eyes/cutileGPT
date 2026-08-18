# SPDX-License-Identifier: Apache-2.0
"""Configuration parsing must reject arithmetic the kernels do not execute."""

import pytest

from cutile_gpt.models.hf_config import UnsupportedArchitecture, parse_config


def minimal_config(activation: str) -> dict:
    return {
        "model_type": "llama",
        "hidden_act": activation,
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 384,
        "vocab_size": 1024,
        "max_position_embeddings": 2048,
        "rope_theta": 10000,
        "rms_norm_eps": 1e-6,
    }


@pytest.mark.parametrize("activation", ["silu", "swish", "silu_and_mul"])
def test_swiglu_aliases_are_accepted(activation):
    assert parse_config(minimal_config(activation)).model_type == "llama"


@pytest.mark.parametrize("activation", ["geglu", "gelu_pytorch_tanh"])
def test_non_silu_gates_are_refused(activation):
    with pytest.raises(UnsupportedArchitecture, match="only SwiGLU"):
        parse_config(minimal_config(activation))
