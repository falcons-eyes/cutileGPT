#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test cutileGPT text generation with GPT-2 tokenizer.

Uses random weights but real tokenization to demonstrate the complete pipeline.
"""

import cupy as cp
from transformers import GPT2Tokenizer
from cutile_gpt.model import CutileGPT, CutileGPTConfig


def test_generation():
    """Test text generation with cutileGPT."""
    print("="*80)
    print("cutileGPT Text Generation Test")
    print("="*80)

    # Load GPT-2 tokenizer
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create cutileGPT model (using random weights for demonstration)
    print("Creating cutileGPT model...")
    config = CutileGPTConfig.gpt2()
    model = CutileGPT(config, use_fused_mlp=False)

    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "The key to understanding quantum mechanics",
        "In the year 2050, humanity will",
    ]

    for prompt in prompts:
        print(f"\n{'='*80}")
        print(f"Prompt: {prompt}")
        print(f"{'='*80}")

        # Tokenize
        encoded = tokenizer.encode(prompt, return_tensors='pt')
        tokens_list = encoded[0].tolist()
        print(f"Tokens ({len(tokens_list)}): {tokens_list}")

        # Convert to CuPy
        idx = cp.array([tokens_list], dtype=cp.int64)

        # Generate
        print("\nGenerating...")
        generated = model.generate(
            idx,
            max_new_tokens=30,
            temperature=0.9,
            top_k=50
        )

        # Decode
        generated_tokens = cp.asnumpy(generated[0]).tolist()
        output_text = tokenizer.decode(generated_tokens)

        print(f"\nGenerated text:")
        print(f"{output_text}")
        print(f"\nGenerated {len(generated_tokens)} total tokens")

    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)


def benchmark_generation():
    """Benchmark generation speed."""
    print("\n" + "="*80)
    print("Generation Benchmark")
    print("="*80)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    config = CutileGPTConfig.gpt_tile_medium()
    model = CutileGPT(config, use_fused_mlp=False)

    prompt = "The future of AI"
    encoded = tokenizer.encode(prompt, return_tensors='pt')
    idx = cp.array([encoded[0].tolist()], dtype=cp.int64)

    # Warmup
    for _ in range(3):
        _ = model.generate(idx, max_new_tokens=10, temperature=0.8)

    # Benchmark
    import time
    iterations = 10
    tokens_per_iter = 20

    print(f"\nGenerating {tokens_per_iter} tokens, {iterations} iterations...")

    start = time.time()
    for _ in range(iterations):
        _ = model.generate(idx, max_new_tokens=tokens_per_iter, temperature=0.8)
    cp.cuda.Stream.null.synchronize()
    end = time.time()

    elapsed = end - start
    tokens_per_sec = (tokens_per_iter * iterations) / elapsed
    ms_per_token = (elapsed * 1000) / (tokens_per_iter * iterations)

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Tokens/sec: {tokens_per_sec:.1f}")
    print(f"  ms/token: {ms_per_token:.2f}")


if __name__ == "__main__":
    test_generation()
    benchmark_generation()
