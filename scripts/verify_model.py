"""Run a checkpoint through cutileGPT and compare it against transformers.

    uv run python scripts/verify_model.py Qwen/Qwen3-0.6B
    uv run python scripts/verify_model.py /path/to/local/dir --no-download

Agreement is judged on argmax rather than on the logits themselves. Both sides
run in bfloat16 and accumulate in a different order, so the values drift by a
fraction of a logit; what matters is whether the model picks the same tokens.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, humans first",
]


def resolve(target: str, allow_download: bool) -> Path:
    path = Path(target)
    if path.is_dir():
        return path
    if not allow_download:
        raise SystemExit(f"{target} is not a local directory and --no-download is set")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        target, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--tokens", type=int, default=16, help="tokens to generate")
    ap.add_argument("--fp32", action="store_true",
                    help="run both sides in float32. Llama-family models carry "
                         "activations around 400, where one bfloat16 ulp is ~2 - "
                         "use this to separate a precision gap from a real one")
    args = ap.parse_args()

    import cupy as cp
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cutile_gpt.models.hf_config import UnsupportedArchitecture
    from cutile_gpt.models.transformer import TransformerLM

    path = resolve(args.model, not args.no_download)

    try:
        model = TransformerLM.from_pretrained(path)
    except UnsupportedArchitecture as exc:
        print(f"{args.model}: unsupported -- {exc}")
        return 2

    print(f"{args.model}\n  {model}")

    tok = AutoTokenizer.from_pretrained(path)
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).cuda().eval()
    if args.fp32:
        model.weights = {k: v.astype(cp.float32) for k, v in model.weights.items()}
        print("  running both sides in float32")

    failures = 0
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            ref = hf(ids.cuda()).logits.float()
        ours = torch.from_dlpack(
            model.forward(cp.asarray(ids.numpy().astype("int32")))).float()

        agree = (ours.argmax(-1) == ref.argmax(-1)).float().mean().item()
        err = (ours - ref).abs().max().item()
        top_ours = [tok.decode([i]) for i in ours[0, -1].topk(3).indices.tolist()]
        top_ref = [tok.decode([i]) for i in ref[0, -1].topk(3).indices.tolist()]

        # Judged on argmax alone. Lower ranks are frequently exact ties in
        # bfloat16 - transformers itself rounds two candidates to the same
        # logit - so ordering among them is a tie-break, not a disagreement.
        ok = agree == 1.0
        failures += not ok

        print(f"  {'ok ' if ok else 'BAD'} {prompt!r:34} "
              f"argmax {agree:.0%}  max|dlogit| {err:.3f}  top3 {top_ours}")
        if top_ours != top_ref:
            print(f"      (transformers ranked {top_ref}; lower ranks tie in bf16)")

    # Generation, with and without the cache, to check they agree.
    ids = cp.asarray(tok(PROMPTS[0], return_tensors="np").input_ids.astype("int32"))
    texts = []
    for use_cache in (True, False):
        cache = model.new_cache(max_seq_len=512) if use_cache else None
        out = ids
        start = time.time()
        for step in range(args.tokens):
            inp = out if (cache is None or step == 0) else out[:, -1:]
            logits = model.forward(inp, cache=cache, last_token_only=True)
            nxt = cp.argmax(logits[:, -1, :], axis=-1).astype(cp.int32).reshape(1, 1)
            out = cp.concatenate([out, nxt], axis=1)
        cp.cuda.Stream.null.synchronize()
        texts.append(tok.decode(cp.asnumpy(out)[0]))
        label = "cached" if use_cache else "uncached"
        print(f"  {label:8} {(time.time() - start) * 1000 / args.tokens:6.1f} ms/token")

    if texts[0] != texts[1]:
        print("  BAD cached and uncached generation diverged")
        failures += 1
    print(f"  -> {texts[0]!r}")

    if failures and not args.fp32:
        print("\n  argmax disagreed in bfloat16. Re-run with --fp32: if that "
              "passes, the gap is precision on large activations, not logic.")
    print(f"\n{'PASS' if not failures else f'FAIL ({failures})'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
