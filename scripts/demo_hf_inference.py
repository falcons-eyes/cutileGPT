#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
HuggingFace GPT-2 Inference Demo with Tile Programming

This demo showcases:
1. Auto-detection of data characteristics
2. Loading real GPT-2 weights from HuggingFace
3. Running inference using our Tile Programming kernels
4. Clear, declarative API usage

Usage:
    python demo_hf_inference.py

Requirements:
    pip install transformers datasets tiktoken
"""

import sys
import os
import time
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import cupy as cp
import numpy as np

# Add parent directory to path for cutile_gpt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# =============================================================================
# Part 1: Data Auto-Detection System
# =============================================================================

@dataclass
class DataProfile:
    """
    Auto-detected data characteristics.

    Tile Programming Philosophy: Know your data before processing.
    """
    # Basic info
    name: str = ""
    source: str = ""  # "huggingface", "local", "custom"

    # Shape analysis
    sample_count: int = 0
    sequence_lengths: List[int] = field(default_factory=list)
    avg_sequence_length: float = 0.0
    max_sequence_length: int = 0
    min_sequence_length: int = 0

    # Vocabulary analysis
    vocab_size: int = 0
    unique_tokens: int = 0
    token_distribution: Dict[int, int] = field(default_factory=dict)

    # Recommended tile configuration
    recommended_tile_m: int = 64
    recommended_tile_n: int = 64
    recommended_batch_size: int = 1

    # Memory estimation
    estimated_memory_mb: float = 0.0

    def __repr__(self):
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    DATA PROFILE                              ║
╠══════════════════════════════════════════════════════════════╣
║ Name: {self.name:<54} ║
║ Source: {self.source:<52} ║
╠══════════════════════════════════════════════════════════════╣
║ Samples: {self.sample_count:<51} ║
║ Sequence Length: min={self.min_sequence_length}, max={self.max_sequence_length}, avg={self.avg_sequence_length:.1f}
║ Vocabulary Size: {self.vocab_size:<43} ║
║ Unique Tokens Used: {self.unique_tokens:<40} ║
╠══════════════════════════════════════════════════════════════╣
║ RECOMMENDED TILE CONFIG (auto-detected):                     ║
║   tile_m: {self.recommended_tile_m:<50} ║
║   tile_n: {self.recommended_tile_n:<50} ║
║   batch_size: {self.recommended_batch_size:<47} ║
║ Est. Memory: {self.estimated_memory_mb:.1f} MB{' ':<40}║
╚══════════════════════════════════════════════════════════════╝
"""


class DataAnalyzer:
    """
    Automatic data analysis and profiling.

    Tile Programming Philosophy: Understand data characteristics
    to optimize tile configurations automatically.
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def analyze(
        self,
        data: Any,
        name: str = "dataset",
        max_samples: int = 1000
    ) -> DataProfile:
        """
        Analyze data and return a profile with recommendations.

        Args:
            data: Dataset or list of samples
            name: Name for the profile
            max_samples: Maximum samples to analyze

        Returns:
            DataProfile with auto-detected characteristics
        """
        profile = DataProfile(name=name)

        # Detect data source and format
        if hasattr(data, '__class__') and 'Dataset' in data.__class__.__name__:
            profile.source = "huggingface"
            samples = self._extract_hf_samples(data, max_samples)
        elif isinstance(data, list):
            profile.source = "list"
            samples = data[:max_samples]
        elif isinstance(data, np.ndarray) or isinstance(data, cp.ndarray):
            profile.source = "array"
            samples = [data] if data.ndim == 1 else list(data)
        else:
            profile.source = "custom"
            samples = list(data)[:max_samples]

        # Tokenize if needed
        tokenized_samples = self._ensure_tokenized(samples)

        # Analyze sequences
        profile.sample_count = len(tokenized_samples)
        profile.sequence_lengths = [len(s) for s in tokenized_samples]
        profile.avg_sequence_length = np.mean(profile.sequence_lengths)
        profile.max_sequence_length = max(profile.sequence_lengths)
        profile.min_sequence_length = min(profile.sequence_lengths)

        # Analyze vocabulary
        all_tokens = []
        for s in tokenized_samples:
            all_tokens.extend(s if isinstance(s, list) else s.tolist())
        profile.unique_tokens = len(set(all_tokens))

        if self.tokenizer:
            profile.vocab_size = self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else len(self.tokenizer)
        else:
            profile.vocab_size = max(all_tokens) + 1 if all_tokens else 50257

        # Compute recommendations
        profile = self._compute_recommendations(profile)

        return profile

    def _extract_hf_samples(self, dataset, max_samples: int) -> List:
        """Extract samples from HuggingFace dataset."""
        samples = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            # Try common column names
            if 'text' in item:
                samples.append(item['text'])
            elif 'content' in item:
                samples.append(item['content'])
            elif 'input_ids' in item:
                samples.append(item['input_ids'])
            else:
                # Take first string column
                for v in item.values():
                    if isinstance(v, str):
                        samples.append(v)
                        break
        return samples

    def _ensure_tokenized(self, samples: List) -> List:
        """Ensure all samples are tokenized."""
        tokenized = []
        for s in samples:
            if isinstance(s, str):
                if self.tokenizer:
                    tokens = self.tokenizer.encode(s)
                else:
                    # Fallback: simple character-level
                    tokens = [ord(c) % 50257 for c in s]
                tokenized.append(tokens)
            elif isinstance(s, (list, np.ndarray)):
                tokenized.append(list(s) if isinstance(s, np.ndarray) else s)
            else:
                tokenized.append([int(s)])
        return tokenized

    def _compute_recommendations(self, profile: DataProfile) -> DataProfile:
        """Compute optimal tile configuration based on data characteristics."""

        # Tile size recommendations based on sequence length
        avg_seq = profile.avg_sequence_length

        if avg_seq <= 128:
            profile.recommended_tile_m = 32
            profile.recommended_tile_n = 32
        elif avg_seq <= 512:
            profile.recommended_tile_m = 64
            profile.recommended_tile_n = 64
        else:
            profile.recommended_tile_m = 128
            profile.recommended_tile_n = 128

        # Batch size based on memory constraints
        # Rough estimate: GPT-2 small needs ~500MB for batch_size=1
        estimated_per_sample_mb = (profile.max_sequence_length * 768 * 4) / (1024 * 1024)
        available_mb = 8000  # Assume 8GB GPU
        profile.recommended_batch_size = max(1, int(available_mb / (estimated_per_sample_mb * 50)))
        profile.recommended_batch_size = min(profile.recommended_batch_size, 32)

        # Memory estimation
        profile.estimated_memory_mb = estimated_per_sample_mb * profile.recommended_batch_size * 10

        return profile


# =============================================================================
# Part 2: HuggingFace GPT-2 Weight Loader
# =============================================================================

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

        # Load model and tokenizer
        self._hf_model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self._tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

        # Convert weights to CuPy
        weights = self._convert_weights()

        print(f"✅ Loaded {self.model_name}: {self._count_params(weights):.1f}M parameters")

        return weights, self._tokenizer

    def _convert_weights(self) -> Dict[str, cp.ndarray]:
        """Convert PyTorch state dict to CuPy arrays."""
        import torch

        sd = self._hf_model.state_dict()
        weights = {}

        for name, param in sd.items():
            # Convert to CuPy
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


# =============================================================================
# Part 3: Tile-Optimized GPT-2 Model
# =============================================================================

class TileGPT2:
    """
    GPT-2 model using Tile Programming API.

    This implementation uses our declarative Tile API for
    maximum clarity and performance.
    """

    def __init__(self, config: dict, weights: Dict[str, cp.ndarray]):
        self.n_layer = config['n_layer']
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.vocab_size = 50257
        self.block_size = 1024

        self.weights = weights
        self._setup_weight_mappings()

    def _setup_weight_mappings(self):
        """Setup easy access to weights with proper transposition.

        HuggingFace GPT-2 weights are (in_features, out_features)
        Our cutile_linear_bias expects (out_features, in_features)
        So we transpose all linear weights.
        """
        # Map HuggingFace weight names to our naming convention
        self.wte = self.weights['transformer.wte.weight']
        self.wpe = self.weights['transformer.wpe.weight']

        def transpose_contiguous(w):
            """Transpose and ensure contiguous."""
            w_t = cp.transpose(w)
            if not w_t.flags.c_contiguous:
                w_t = cp.ascontiguousarray(w_t)
            return w_t

        self.blocks = []
        for i in range(self.n_layer):
            # HuggingFace: weight is (in, out), we need (out, in)
            block = {
                'ln1_w': self.weights[f'transformer.h.{i}.ln_1.weight'],
                'ln1_b': self.weights[f'transformer.h.{i}.ln_1.bias'],
                # Attention weights need transpose
                'attn_w': transpose_contiguous(self.weights[f'transformer.h.{i}.attn.c_attn.weight']),
                'attn_b': self.weights[f'transformer.h.{i}.attn.c_attn.bias'],
                'attn_proj_w': transpose_contiguous(self.weights[f'transformer.h.{i}.attn.c_proj.weight']),
                'attn_proj_b': self.weights[f'transformer.h.{i}.attn.c_proj.bias'],
                'ln2_w': self.weights[f'transformer.h.{i}.ln_2.weight'],
                'ln2_b': self.weights[f'transformer.h.{i}.ln_2.bias'],
                # MLP weights need transpose
                'mlp_fc_w': transpose_contiguous(self.weights[f'transformer.h.{i}.mlp.c_fc.weight']),
                'mlp_fc_b': self.weights[f'transformer.h.{i}.mlp.c_fc.bias'],
                'mlp_proj_w': transpose_contiguous(self.weights[f'transformer.h.{i}.mlp.c_proj.weight']),
                'mlp_proj_b': self.weights[f'transformer.h.{i}.mlp.c_proj.bias'],
            }
            self.blocks.append(block)

        self.ln_f_w = self.weights['transformer.ln_f.weight']
        self.ln_f_b = self.weights['transformer.ln_f.bias']

        print(f"  Weights transposed: attn_w shape = {self.blocks[0]['attn_w'].shape}")
        print(f"  Expected: (2304, 768) = (3*n_embd, n_embd)")

    def forward(self, idx: cp.ndarray) -> cp.ndarray:
        """
        Forward pass using Tile Programming kernels.

        Args:
            idx: Token indices (batch, seq_len)

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        from cutile_gpt.kernels import (
            cutile_layer_norm, cutile_linear_bias, cutile_gelu,
            cutile_causal_attention
        )

        batch, seq_len = idx.shape

        # Embeddings
        tok_emb = self.wte[idx]  # (batch, seq, n_embd)
        pos = cp.arange(seq_len, dtype=cp.int64)
        pos_emb = self.wpe[pos]  # (seq, n_embd)
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.blocks:
            # Attention
            x_norm = cutile_layer_norm(x, block['ln1_w'], block['ln1_b'])

            # QKV projection
            qkv = cutile_linear_bias(x_norm, block['attn_w'], block['attn_b'])

            # Split Q, K, V
            q, k, v = cp.split(qkv, 3, axis=-1)

            # Reshape for multi-head attention
            head_dim = self.n_embd // self.n_head
            q = cp.transpose(q.reshape(batch, seq_len, self.n_head, head_dim), (0, 2, 1, 3))
            k = cp.transpose(k.reshape(batch, seq_len, self.n_head, head_dim), (0, 2, 1, 3))
            v = cp.transpose(v.reshape(batch, seq_len, self.n_head, head_dim), (0, 2, 1, 3))

            # Attention
            attn_out = cutile_causal_attention(q, k, v, self.n_head)

            # Reshape back
            attn_out = cp.transpose(attn_out, (0, 2, 1, 3))
            attn_out = cp.ascontiguousarray(attn_out).reshape(batch, seq_len, self.n_embd)

            # Output projection
            attn_out = cutile_linear_bias(attn_out, block['attn_proj_w'], block['attn_proj_b'])
            x = x + attn_out

            # MLP
            x_norm = cutile_layer_norm(x, block['ln2_w'], block['ln2_b'])
            hidden = cutile_linear_bias(x_norm, block['mlp_fc_w'], block['mlp_fc_b'])
            hidden = cutile_gelu(hidden)
            mlp_out = cutile_linear_bias(hidden, block['mlp_proj_w'], block['mlp_proj_b'])
            x = x + mlp_out

        # Final layer norm
        x = cutile_layer_norm(x, self.ln_f_w, self.ln_f_b)

        # LM head (tied weights)
        logits = x @ self.wte.T

        return logits

    def generate(
        self,
        idx: cp.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> cp.ndarray:
        """
        Generate text autoregressively.

        Args:
            idx: Starting tokens (batch, seq)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering

        Returns:
            Generated tokens (batch, seq + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx if idx.shape[1] <= self.block_size else idx[:, -self.block_size:]

            # Forward
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v = cp.partition(logits, -top_k, axis=-1)[:, -top_k]
                logits = cp.where(logits < v[:, None], float('-inf'), logits)

            # Softmax
            probs = cp.exp(logits - cp.max(logits, axis=-1, keepdims=True))
            probs = probs / cp.sum(probs, axis=-1, keepdims=True)

            # Sample
            batch_size = probs.shape[0]
            idx_next = cp.zeros((batch_size, 1), dtype=cp.int64)
            for b in range(batch_size):
                cumsum = cp.cumsum(probs[b])
                r = cp.random.rand()
                idx_next[b, 0] = cp.searchsorted(cumsum, r)

            idx = cp.concatenate([idx, idx_next], axis=1)

        return idx


# =============================================================================
# Part 4: Demo with Fluent Tile API
# =============================================================================

def demo_with_tile_api():
    """
    Demonstrate the declarative Tile API with real GPT-2.
    """
    from cutile_gpt.api import tile, configure_tiles, TileConfig

    print("\n" + "="*70)
    print("      TILE API DEMONSTRATION - Declarative Data Flow")
    print("="*70 + "\n")

    # Create sample data
    batch, seq, n_embd = 2, 64, 768
    n_hidden = 4 * n_embd

    x = cp.random.randn(batch, seq, n_embd, dtype=cp.float32)
    w_fc = cp.random.randn(n_hidden, n_embd, dtype=cp.float32) * 0.02
    b_fc = cp.random.randn(n_hidden, dtype=cp.float32) * 0.02
    w_proj = cp.random.randn(n_embd, n_hidden, dtype=cp.float32) * 0.02
    b_proj = cp.random.randn(n_embd, dtype=cp.float32) * 0.02
    ln_w = cp.ones(n_embd, dtype=cp.float32)
    ln_b = cp.zeros(n_embd, dtype=cp.float32)

    print("1. Building computation graph with Tile API:")
    print("-" * 50)

    # Build declarative computation graph
    op = (
        tile(x, "hidden_states")
            .declare_shape(batch=batch, seq=seq, dim=n_embd)
            .linear(w_fc, b_fc, out_features=n_hidden)
            .gelu()
            .linear(w_proj, b_proj, out_features=n_embd)
            .layernorm(ln_w, ln_b)
    )

    print(op.describe())

    print("\n2. Executing with custom tile configuration:")
    print("-" * 50)

    config = configure_tiles(tile_m=64, tile_n=64, tile_k=32)
    print(f"Tile Config: tm={config.tile_m}, tn={config.tile_n}, tk={config.tile_k}")

    start = time.time()
    result = op.execute()
    elapsed = time.time() - start

    print(f"Result shape: {result.shape}")
    print(f"Execution time: {elapsed*1000:.2f} ms")
    print("\n✅ Tile API demonstration complete!")


# =============================================================================
# Part 5: Main Demo
# =============================================================================

def main():
    """Main demo entry point."""

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║          cutileGPT - Tile Programming Inference Demo                 ║
║                                                                      ║
║  This demo showcases:                                                ║
║    1. Auto-detection of data characteristics                         ║
║    2. Loading real GPT-2 weights from HuggingFace                    ║
║    3. Running inference using Tile Programming kernels               ║
║    4. Declarative, explicit data flow                                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # Step 1: Load HuggingFace GPT-2
    print("\n" + "="*70)
    print("      STEP 1: Loading GPT-2 from HuggingFace")
    print("="*70)

    loader = HFWeightLoader('gpt2')
    weights, tokenizer = loader.load()

    # Step 2: Load and analyze dataset
    print("\n" + "="*70)
    print("      STEP 2: Loading and Analyzing Dataset")
    print("="*70)

    try:
        from datasets import load_dataset
        print("\nLoading wikitext-2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

        analyzer = DataAnalyzer(tokenizer)
        profile = analyzer.analyze(dataset, name="wikitext-2-raw-v1", max_samples=100)
        print(profile)

    except ImportError:
        print("\n⚠️  'datasets' library not installed. Using sample text instead.")
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In a hole in the ground there lived a hobbit.",
            "It was the best of times, it was the worst of times.",
        ]
        analyzer = DataAnalyzer(tokenizer)
        profile = analyzer.analyze(sample_texts, name="sample_texts")
        print(profile)

    # Step 3: Create model and run inference
    print("\n" + "="*70)
    print("      STEP 3: Running Inference with Tile Kernels")
    print("="*70)

    model = TileGPT2(loader.config, weights)

    # Test prompts
    prompts = [
        "The future of artificial intelligence",
        "Once upon a time in a land far away",
        "The key to successful machine learning is",
    ]

    print("\nGenerating text with cutile GPT-2:\n")

    for prompt in prompts:
        print(f"Prompt: \"{prompt}\"")
        print("-" * 60)

        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors='np')
        input_ids = cp.asarray(input_ids)

        # Generate
        start = time.time()
        output_ids = model.generate(input_ids, max_new_tokens=30, temperature=0.8, top_k=50)
        elapsed = time.time() - start

        # Decode
        output_text = tokenizer.decode(cp.asnumpy(output_ids[0]))

        print(f"Generated: {output_text}")
        print(f"Time: {elapsed*1000:.1f} ms | Tokens/sec: {30/elapsed:.1f}")
        print()

    # Step 4: Demonstrate Tile API
    demo_with_tile_api()

    # Summary
    print("\n" + "="*70)
    print("      SUMMARY: Tile Programming Benefits")
    print("="*70)
    print("""
✅ DECLARATIVE: Specify WHAT, not HOW
   - No thread management
   - No manual synchronization
   - Clean, readable code

✅ AUTO-OPTIMIZATION: Compiler handles optimization
   - Tile size selection
   - Memory layout
   - Instruction scheduling

✅ EXPLICIT DATA FLOW: Know your data at every step
   - Shape declarations
   - Type checking
   - Profile analysis

✅ COMPOSABLE: Operations combine naturally
   - Method chaining
   - Graph building
   - Lazy execution

This is the future of GPU programming!
""")


if __name__ == "__main__":
    main()
