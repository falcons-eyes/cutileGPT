#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test cutileGPT with real GPT-2 weights and actual text generation.

This script:
1. Loads pretrained GPT-2 weights from HuggingFace
2. Converts them to cutileGPT format
3. Performs text generation with actual prompts
4. Compares output with PyTorch minGPT
"""

import sys
import cupy as cp
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Add external/minGPT to path
sys.path.insert(0, 'external/minGPT')
from mingpt.model import GPT as minGPT
from mingpt.bpe import BPETokenizer

from cutile_gpt.model import CutileGPT, CutileGPTConfig


def load_gpt2_weights(model_type='gpt2'):
    """
    Load GPT-2 weights from HuggingFace and convert to cutileGPT.

    Args:
        model_type: One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

    Returns:
        Tuple of (mingpt_model, cutile_model, tokenizer)
    """
    print(f"Loading {model_type} from HuggingFace...")

    # Load minGPT with pretrained weights
    mingpt_model = minGPT.from_pretrained(model_type)
    mingpt_model.eval()

    # Create cutileGPT config matching GPT-2
    if model_type == 'gpt2':
        config = CutileGPTConfig.gpt2()
    else:
        # For other sizes, create custom config
        model_configs = {
            'gpt2-medium': {'n_layer': 24, 'n_head': 16, 'n_embd': 1024},
            'gpt2-large': {'n_layer': 36, 'n_head': 20, 'n_embd': 1280},
            'gpt2-xl': {'n_layer': 48, 'n_head': 25, 'n_embd': 1600},
        }
        params = model_configs[model_type]
        config = CutileGPTConfig(
            vocab_size=50257,
            block_size=1024,
            n_layer=params['n_layer'],
            n_head=params['n_head'],
            n_embd=params['n_embd']
        )

    print(f"Creating cutileGPT model: {config.n_layer} layers, {config.n_embd} dims, {config.n_head} heads")
    cutile_model = CutileGPT(config)

    # Transfer weights from minGPT to cutileGPT
    print("Transferring weights from PyTorch to CuPy...")
    cutile_model.load_from_mingpt(mingpt_model)

    # Load tokenizer
    tokenizer = BPETokenizer()

    return mingpt_model, cutile_model, tokenizer


def test_generation(mingpt_model, cutile_model, tokenizer, prompt, max_new_tokens=20):
    """
    Test text generation with both models and compare.

    Args:
        mingpt_model: PyTorch minGPT model
        cutile_model: CuPy cutileGPT model
        tokenizer: BPE tokenizer
        prompt: Text prompt
        max_new_tokens: Number of tokens to generate
    """
    print(f"\n{'='*80}")
    print(f"Prompt: {prompt}")
    print(f"{'='*80}")

    # Tokenize
    tokens = tokenizer(prompt)
    print(f"Tokenized ({len(tokens)} tokens): {tokens[:10].tolist()}...")

    # PyTorch minGPT generation
    print("\n[PyTorch minGPT]")
    with torch.no_grad():
        mingpt_model.eval()
        tokens_torch = tokens.unsqueeze(0)  # (1, seq_len)
        generated_torch = mingpt_model.generate(
            tokens_torch,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_k=40
        )[0]

    output_torch = tokenizer.decode(generated_torch.cpu())
    print(f"Generated: {output_torch}")

    # CuPy cutileGPT generation
    print("\n[CuPy cutileGPT]")
    tokens_cupy = cp.asarray(tokens.cpu().numpy()).astype(cp.int64).reshape(1, -1)
    generated_cupy = cutile_model.generate(
        tokens_cupy,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=40
    )

    output_cupy = tokenizer.decode(torch.from_numpy(cp.asnumpy(generated_cupy[0])))
    print(f"Generated: {output_cupy}")

    # Compare forward pass logits
    print("\n[Logits Comparison]")
    with torch.no_grad():
        logits_torch, _ = mingpt_model(tokens_torch)

    logits_cupy, _ = cutile_model(tokens_cupy)
    logits_cupy_np = cp.asnumpy(logits_cupy)

    diff = abs(logits_torch.cpu().numpy() - logits_cupy_np).max()
    print(f"Max logits difference: {diff:.6e}")

    if diff < 1e-3:
        print("✓ Logits match within tolerance!")
    else:
        print("⚠ Logits differ significantly")

    return output_torch, output_cupy


def main():
    """Main test function."""
    # Test with GPT-2 base model
    model_type = 'gpt2'

    print("="*80)
    print("cutileGPT Real GPT-2 Test")
    print("="*80)

    mingpt_model, cutile_model, tokenizer = load_gpt2_weights(model_type)

    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "The key to understanding quantum mechanics",
        "In the year 2050, humanity will",
    ]

    for prompt in prompts:
        test_generation(mingpt_model, cutile_model, tokenizer, prompt, max_new_tokens=30)

    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)


if __name__ == "__main__":
    main()
