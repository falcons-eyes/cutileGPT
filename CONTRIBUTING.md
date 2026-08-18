# Contributing to cutileGPT

Thanks for taking a look. This is an experimental project exploring whether a
full GPT can be written declaratively with NVIDIA cuTile instead of hand-tuned
CUDA, so bug reports and benchmark numbers from hardware we do not have are
genuinely useful contributions.

## Hardware reality check

cuTile's `tileiras` compiler currently targets **Blackwell only**:

| | Requirement |
|---|---|
| GPU | `sm_100` (B200/GB200) or `sm_120` (GB10, RTX 50 series) |
| CUDA Toolkit | 13.1+ |
| NVIDIA Driver | r580+ |
| Python | 3.13+ |

Hopper (`sm_90`) and earlier cannot run the kernels yet. Upstream describes this
as a temporary restriction.

**You do not need a GPU to contribute.** Documentation, the visualization
scripts, packaging, and CI all run fine without one, and CI itself is GPU-free.

## Setup

```bash
git clone --recursive https://github.com/falcons-eyes/cutileGPT.git
cd cutileGPT
uv sync --all-extras
```

If you cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

## Before you open a PR

CI runs these; running them locally first is faster than waiting on it.

```bash
uvx ruff check .                    # lint - must be clean
uv lock --check                     # lockfile matches pyproject.toml
uv build                            # sdist + wheel build
uv run python scripts/check_dependencies.py   # no undeclared imports
uv run python scripts/check_imports.py        # imports without optional extras
```

The last two exist because of a real regression: v0.2.0 shipped without
`cuda-tile` in `requires-dist`, so `pip install cutile-gpt` resolved cleanly and
then died on `import cuda.tile`. If you add an import to `cutile_gpt/`, add the
distribution to `[project.dependencies]` too - or import it lazily inside the
function that needs it, the way `hf_loader.py` handles `transformers`.

Tests under `tests/` need a Blackwell GPU and are not run by CI:

```bash
uv run pytest
```

`tests/test_dtypes.py` is the exception - its dtype and loader cases run without
a GPU and skip the kernel cases automatically.

## A note on dtypes

Kernels take the dtype of the arrays passed to them; none of them hardcode
float32. Keep it that way - accumulate in fp32 internally and store back with
`.astype(Out.dtype)`, and a kernel written for float32 handles bfloat16 for
free. That is what makes porting to a real checkpoint cheap.

Load weights with `cp.from_dlpack(tensor.contiguous().cuda())`, never
`cp.asarray(tensor.numpy())` - numpy has no bfloat16, and every current
open-weight model ships in it.

## Reporting a bug

Please include GPU model, `nvidia-smi` driver version, CUDA Toolkit version, and
the output of `python -c "import cutile_gpt; print(cutile_gpt.__version__)"`.
Compilation errors from `tileiras` are much easier to act on with the full
traceback attached.

## Benchmark contributions

Published numbers come from a single NVIDIA GB10, which makes them impossible
for most people to reproduce. Results from other Blackwell parts are welcome -
run `uv run python scripts/profile_performance.py` and open an issue with the
JSON output plus your hardware details.

## Style

`ruff` with the settings in `pyproject.toml` is the whole style guide. Match the
surrounding code; kernels are written to be read as teaching material, so prefer
a clear shape over a clever one.

## License

Contributions are licensed under [Apache-2.0](LICENSE), matching the project.
