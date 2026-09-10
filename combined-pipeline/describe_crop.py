"""Describe the crop produced by capture_and_detect.py using llama-mtmd-cli, log
timing + the description as JSON.

Timing-parsing logic (regexes, subprocess approach) duplicated from
vlm-benchmarks/benchmark_llama_cpp_vlm.py rather than imported -- this script runs
inside a different container (llama_cpp) with a different working directory than
vlm-benchmarks/, and duplicating a few lines keeps it runnable standalone rather
than needing PYTHONPATH wiring across the two directories (same tradeoff already
made between llm-benchmarks/run_all.sh and run_tensorrt_edgellm.sh). If llama.cpp's
log wording changes again, see vlm-benchmarks/benchmark_llama_cpp_vlm.py's
docstring for how that was actually diagnosed last time (real captured logs, not
guessed from upstream source).

Run inside the llama_cpp jetson-container, after capture_and_detect.py has
written crop.jpg + detection.json to the shared .cache dir:
    python3 describe_crop.py --model gemma-3-4b-it-Q4_K_M.gguf --mmproj mmproj-model-f16.gguf

No warmup here, unlike capture_and_detect.py's vision stage: llama-mtmd-cli is a
fresh subprocess every call, so model loading (parsed below as load_ms) happens
cold on every single invocation regardless of a warmup loop -- there's no
in-process state a warmup could exercise ahead of time. That's an honest
limitation of this one-shot-CLI design, not something to paper over; a real
deployment would run a persistent llama-server instead, loading the model once
and serving many requests. Reporting load_ms separately (rather than letting it
hide inside prefill/decode) is what actually makes that cost visible.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

RE_PREPROCESS = re.compile(r"image/slice encoded in (\d+) ms")
RE_LOAD = re.compile(r"load time = \s*([\d.]+) ms")
RE_PREFILL = re.compile(r"prompt eval time = \s*([\d.]+) ms / \s*(\d+) tokens")
RE_DECODE = re.compile(r"(?<!prompt )eval time = \s*([\d.]+) ms / \s*(\d+) runs.*?([\d.]+) tokens per second")


def build_prompt(detection: dict | None) -> str:
    if detection is None:
        return "Describe what you see in this image in detail."
    return f"Describe this {detection['class_name']} in detail. What is it, and what does it look like?"


def describe(model: str, mmproj: str, image: str, prompt: str, n_predict: int):
    proc = subprocess.run(
        [
            "llama-mtmd-cli",
            "-m", model,
            "--mmproj", mmproj,
            "--image", image,
            "-p", prompt,
            "-n", str(n_predict),
            "-ngl", "999",
        ],
        capture_output=True, text=True, check=True,
    )
    log = proc.stderr

    # llama-mtmd-cli prints a startup line ("main: loading model: <path>") to
    # stdout rather than stderr, ahead of the actual generated text -- confirmed
    # against a real run (see notes/vlm-benchmarks.md for this project's general
    # rule of checking real captured output over guessing log formats). Strip any
    # leading blank lines and "word: ..."-style log lines before treating the rest
    # as the description.
    lines = proc.stdout.split("\n")
    while lines and (not lines[0].strip() or re.match(r"^\w+: ", lines[0])):
        lines.pop(0)
    text = "\n".join(lines).strip()

    preprocess_matches = RE_PREPROCESS.findall(log)
    load_m = RE_LOAD.search(log)
    prefill_m = RE_PREFILL.search(log)
    decode_m = RE_DECODE.search(log)
    if not (load_m and prefill_m and decode_m):
        raise RuntimeError(f"could not parse timing from llama-mtmd-cli output:\n{log[-3000:]}")

    preprocessing_ms = sum(float(m) for m in preprocess_matches) if preprocess_matches else 0.0
    prefill_ms = float(prefill_m.group(1))

    return {
        "description": text,
        "vlm_load_ms": float(load_m.group(1)),  # cold every call -- see module docstring
        "preprocessing_ms": preprocessing_ms,
        "prefill_ms": prefill_ms,
        # NOT measured via streaming -- subprocess.run() blocks until the whole
        # process exits, so this script never observes token-by-token arrival even
        # though llama-mtmd-cli streams internally to its own stdout. Still an
        # accurate number: the first generated token becomes available right when
        # prefill's last step finishes (sampling it is a negligible cost on top),
        # so preprocessing+prefill IS time-to-first-token, not an approximation of it.
        "ttft_ms": preprocessing_ms + prefill_ms,
        "decode_ms": float(decode_m.group(1)),
        "tokens_per_sec": float(decode_m.group(3)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the .gguf model file")
    ap.add_argument("--mmproj", required=True, help="path to the .gguf multimodal projector file")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--cache-dir", default="/workspace/combined-pipeline/.cache",
                     help="where capture_and_detect.py wrote crop.jpg / detection.json")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    detection_meta = json.loads((cache_dir / "detection.json").read_text())
    prompt = build_prompt(detection_meta["detection"])

    t_start = time.perf_counter()
    stats = describe(args.model, args.mmproj, str(cache_dir / "crop.jpg"), prompt, args.n_predict)
    vlm_wall_ms = (time.perf_counter() - t_start) * 1000

    result = {
        "prompt": prompt,
        "detection": detection_meta["detection"],
        "capture_ms": detection_meta["capture_ms"],
        "vision_load_ms": detection_meta["vision_load_ms"],
        "vision_cold_inference_ms": detection_meta["vision_cold_inference_ms"],
        "vision_inference_ms": detection_meta["vision_inference_ms"],  # steady-state, post-warmup
        "vlm_wall_ms": vlm_wall_ms,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **stats,
    }
    # end-to-end: camera frame in, text out, everything included -- distinct
    # from either model's standalone latency. This is a genuinely "cold" number
    # every time by construction (see module docstring: the VLM stage can't be
    # warmed up the way the vision stage was), so it's the honest worst-case,
    # not a bug.
    result["end_to_end_ms"] = (
        detection_meta["capture_ms"] + detection_meta["vision_inference_ms"] + vlm_wall_ms
    )
    # steady-state compute only: excludes camera capture and both stages' one-time
    # model-load cost -- what repeated frames through an already-running pipeline
    # would actually cost, as distinct from this one-shot script's necessarily-cold
    # total above. Loading is real cost, just not *per-frame* cost in a persistent
    # deployment (llama-server, a long-lived vision process), which this one-shot
    # design doesn't attempt to be.
    result["compute_only_ms"] = (
        detection_meta["vision_inference_ms"]
        + (stats["preprocessing_ms"] or 0.0)
        + stats["prefill_ms"]
        + stats["decode_ms"]
    )

    out_path = cache_dir / "result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
