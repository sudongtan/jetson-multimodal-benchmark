"""Measure image-preprocessing, prefill, and decode latency for a GGUF VLM via
llama-mtmd-cli, log as JSON.

Shells out to `llama-mtmd-cli` (llama.cpp's own multimodal CLI, confirmed present
and working in the Phase 2 `llama_cpp` jetson-container) rather than using
llama-cpp-python's bindings, since multimodal/mmproj support was verified
against the CLI directly, not the Python API.

Parses three timing lines from the CLI's own stderr output. The first version of
this script guessed the preprocessing line's wording from llama.cpp's `master`
branch source ("mtmd batch encoding done in N ms") -- wrong, because this
container runs an older pinned build (dustynv/llama_cpp:b5283-r36.4-cu128-24.04)
whose actual wording is different. Confirmed instead against this build's real
captured output (see results/vlm-benchmarks/logs/):
  - "image/slice encoded in N ms"  -- image preprocessing (clip/vision encode).
    Printed once per image slice, so multiple lines are summed for models that
    tile an image into several slices. This runs on a separate (clip) context
    from the main llama_context, so it is NOT included in the prompt-eval timing
    below -- the split is real, not an approximation.
  - "prompt eval time = N ms / N tokens (... tokens per second)"  -- text prefill.
  - "eval time = N ms / N runs (... tokens per second)"  -- decode/generation.
    Regex excludes the "prompt eval" line via a negative lookbehind, since
    "eval time" is a substring of it.
If llama.cpp's wording changes again in a future container update, check the raw
per-run logs under results/vlm-benchmarks/logs/ (written by this script) rather
than guessing from upstream source a third time.

Run inside the llama_cpp jetson-container:
    python3 benchmark_llama_cpp_vlm.py --model gemma-3-4b-it-Q4_K_M.gguf \\
        --mmproj mmproj-model-f16.gguf --image cats.jpg \\
        --quant Q4_K_M --quant-class q4
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

PROMPT = "Describe this image in one sentence."

# See module docstring for where each of these comes from in this build's real output.
RE_PREPROCESS = re.compile(r"image/slice encoded in (\d+) ms")
RE_PREFILL = re.compile(r"prompt eval time = \s*([\d.]+) ms / \s*(\d+) tokens")
RE_DECODE = re.compile(r"(?<!prompt )eval time = \s*([\d.]+) ms / \s*(\d+) runs.*?([\d.]+) tokens per second")


def run_once(model: str, mmproj: str, image: str, prompt: str, n_predict: int, log_dir: Path | None, run_idx: int):
    proc = subprocess.run(
        [
            "llama-mtmd-cli",
            "-m", model,
            "--mmproj", mmproj,
            "--image", image,
            "-p", prompt,
            "-n", str(n_predict),
            "-ngl", "999",  # offload all model layers to GPU
        ],
        capture_output=True, text=True, check=True,
    )
    log = proc.stderr  # llama.cpp logs (including the perf lines) go to stderr

    # Always keep the raw log -- the exact print format is version-specific (this
    # project got burned once already assuming upstream master's source matched
    # this container's actual pinned build), so when a regex below fails to match,
    # this is what to grep instead of guessing again.
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"run{run_idx}.log").write_text(log)

    preprocess_matches = RE_PREPROCESS.findall(log)  # one per image slice; sum if tiled
    prefill_m = RE_PREFILL.search(log)
    decode_m = RE_DECODE.search(log)

    if not (prefill_m and decode_m):
        raise RuntimeError(f"could not parse timing from llama-mtmd-cli output:\n{log[-3000:]}")

    preprocessing_ms = sum(float(m) for m in preprocess_matches) if preprocess_matches else None
    prefill_ms = float(prefill_m.group(1))
    prefill_tokens = int(prefill_m.group(2))
    decode_ms = float(decode_m.group(1))
    decode_tokens = int(decode_m.group(2))
    decode_tokens_per_sec = float(decode_m.group(3))

    return {
        "preprocessing_ms": preprocessing_ms,
        "prefill_ms": prefill_ms,
        "prefill_tokens": prefill_tokens,
        # end-to-end time to first generated token: image encode + text prefill.
        "ttft_ms": (preprocessing_ms or 0.0) + prefill_ms,
        "decode_ms": decode_ms,
        "n_predict": decode_tokens,
        "tokens_per_sec": decode_tokens_per_sec,
    }


def benchmark(model: str, mmproj: str, image: str, prompt: str, n_predict: int, runs: int, warmup: int,
              log_dir: Path | None = None):
    samples = []
    for run in range(warmup + runs):
        result = run_once(model, mmproj, image, prompt, n_predict, log_dir, run)
        if run >= warmup:
            samples.append(result)

    def avg(key):
        vals = [s[key] for s in samples if s[key] is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "preprocessing_ms": avg("preprocessing_ms"),
        "prefill_ms": avg("prefill_ms"),
        "ttft_ms": avg("ttft_ms"),
        "decode_ms": avg("decode_ms"),
        "tokens_per_sec": avg("tokens_per_sec"),
        "n_predict": samples[0]["n_predict"],
        "prefill_tokens": samples[0]["prefill_tokens"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the .gguf model file")
    ap.add_argument("--mmproj", required=True, help="path to the .gguf multimodal projector file")
    ap.add_argument("--image", required=True, help="path to a test image")
    ap.add_argument("--quant", required=True, help="e.g. Q4_K_M/Q8_0/f16 (llama.cpp's own tag)")
    ap.add_argument("--quant-class", required=True, choices=["q4", "q8", "fp16"])
    ap.add_argument("--model-name", default="gemma-3-4b-it")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out-dir", default="/workspace/results/vlm-benchmarks")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    log_dir = out_dir / "logs" / f"{args.model_name}_{args.quant}"
    stats = benchmark(args.model, args.mmproj, args.image, args.prompt, args.n_predict, args.runs, args.warmup,
                       log_dir=log_dir)

    result = {
        "model": args.model_name,
        "framework": "llama_cpp",
        "quantization": args.quant,
        "quant_class": args.quant_class,
        "model_size_mb": Path(args.model).stat().st_size / 1e6,
        "peak_memory_mb": None,  # filled in by scripts/tegrastats_summary.py on the host
        "avg_power_w": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **stats,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"llama_cpp_{args.model_name}_{args.quant}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
