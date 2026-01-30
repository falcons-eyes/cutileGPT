# SPDX-License-Identifier: Apache-2.0
"""
HuggingFace Weight Loader

Utilities for loading GPT-2 weights from HuggingFace transformers.
"""

from typing import Dict, Tuple, Any
import cupy as cp


class HFWeightLoader:
    """
    Load GPT-2 weights from HuggingFace and convert to CuPy.

    Tile Programming Philosophy: Clean separation between
    weight loading and computation.
    """

    SUPPORTED_MODELS = {
        'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
        'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
        'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
        'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
    }

    def __init__(self, model_name: str = 'gpt2'):
        self.model_name = model_name
        self.config = self.SUPPORTED_MODELS.get(model_name, self.SUPPORTED_MODELS['gpt2'])
        self._hf_model = None
        self._tokenizer = None

    def load(self) -> Tuple[Dict[str, cp.ndarray], Any]:
        """
        Load GPT-2 model and tokenizer from HuggingFace.

        Returns:
            Tuple of (weights_dict, tokenizer)
        """
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        print(f"Loading {self.model_name} from HuggingFace...")

        self._hf_model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self._tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

        weights = self._convert_weights()

        print(f"✅ Loaded {self.model_name}: {self._count_params(weights):.1f}M parameters")

        return weights, self._tokenizer

    def _convert_weights(self) -> Dict[str, cp.ndarray]:
        """Convert PyTorch state dict to CuPy arrays."""
        sd = self._hf_model.state_dict()
        weights = {}

        for name, param in sd.items():
            np_array = param.detach().cpu().numpy()
            weights[name] = cp.asarray(np_array)

        return weights

    def _count_params(self, weights: Dict[str, cp.ndarray]) -> float:
        """Count total parameters in millions."""
        total = sum(w.size for w in weights.values())
        return total / 1e6

    @property
    def tokenizer(self):
        return self._tokenizer
