# cutileGPT Profiling Summary

## Performance Benchmarks

Successfully profiled 6 different model configurations:

| Config | Layers | Embedding | Mean (ms) | Throughput (tokens/sec) |
|--------|--------|-----------|-----------|------------------------|
| gpt_nano | 3 | 48 | 1.27 | 201,203 |
| gpt_micro | 4 | 128 | 3.54 | 72,297 |
| gpt_mini | 6 | 192 | 10.88 | 23,525 |
| **gpt_tile_small** | 4 | 64 | 1.52 | **168,623** |
| **gpt_tile_medium** | 6 | 128 | 4.65 | **55,002** |
| **gpt_tile_large** | 8 | 256 | 25.59 | **10,006** |

## Text Generation Test

✅ Successfully tested with GPT-2 tokenizer
- Tokenization: Working
- Generation pipeline: Working
- Performance: ~92 tokens/sec (gpt_tile_medium)
- Time per token: ~10.89 ms

## NVIDIA Profiling

### Nsight Systems (nsys)
- ✅ Profile generated: `profiling_results/cutile_nsys.nsys-rep`
- Output: System-wide timeline and stats
- Note: CUDA trace data not captured (GB10/Blackwell compatibility)

### Nsight Compute (ncu)
- Not yet executed (can provide detailed kernel metrics)

## Files Created

- `profile_performance.py` - Comprehensive performance profiling
- `test_text_generation.py` - Text generation with GPT-2 tokenizer
- `test_gpt2_real.py` - Real GPT-2 weight loading (needs minGPT fix)
- `scripts/run_nsys_profile.sh` - Nsight Systems profiling script
- `scripts/run_ncu_profile.sh` - Nsight Compute profiling script

## Next Steps

1. **View nsys results**: 
   ```bash
   nsys stats profiling_results/cutile_nsys.nsys-rep
   ```

2. **Run ncu for kernel-level analysis**:
   ```bash
   ./scripts/run_ncu_profile.sh
   ```

3. **Compare with PyTorch** (already in compare.py):
   ```bash
   uv run python cutile_gpt/compare.py --benchmark --model tile-medium
   ```
