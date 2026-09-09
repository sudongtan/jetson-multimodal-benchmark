"""Measure TTFT and decode tokens/sec for a TensorRT-Edge-LLM engine, log as JSON.

Shells out to the official `llm_inference` binary with --dumpProfile/--profileOutputFile
(NVIDIA/TensorRT-Edge-LLM v0.5.0's own profiler -- see examples/utils/profileFormatter.cpp)
rather than timing it ourselves; it already reports prefill (~TTFT), generation tokens/sec,
and peak memory in bytes. Run inside the tensorrt_edgellm jetson-container, after building
the engine with build_engine_edgellm.sh:
    python3 benchmark_tensorrt_edgellm.py --engine-dir /workspace/.cache/edgellm/engines/int4_awq \\
        --quant int4_awq --quant-class q4
"""
import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

PROMPT = "Explain what a transformer neural network is, in simple terms."


def benchmark(engine_dir: str, n_predict: int, runs: int, warmup: int):
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.json"
        profile_path = Path(tmp) / "profile.json"

        input_path.write_text(json.dumps({
            "batch_size": 1,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 40,
            "max_generate_length": n_predict,
            # one request per timed run; --warmup reuses the first request separately.
            "requests": [{"messages": [{"role": "user", "content": PROMPT}]}] * runs,
        }))

        subprocess.run(
            [
                "llm_inference",
                "--engineDir", engine_dir,
                "--inputFile", str(input_path),
                "--warmup", str(warmup),
                "--dumpProfile",
                "--profileOutputFile", str(profile_path),
            ],
            check=True,
        )

        profile = json.loads(profile_path.read_text())

    return {
        # prefill = time to process the prompt before the first output token, i.e. TTFT
        # for a single, non-batched request.
        "ttft_ms": profile["prefill"]["average_time_per_run_ms"],
        "tokens_per_sec": profile["generation"]["tokens_per_second"],
        "n_predict": n_predict,
        "input_tokens": profile["prefill"]["computed_tokens"],
        # per-process, from the tool's own profiler -- more precise than the system-wide
        # tegrastats sample used for the other frameworks, kept here for reference.
        "peak_memory_mb_edgellm": profile["peak_unified_memory_bytes"] / 1e6,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", required=True, help="engine dir from build_engine_edgellm.sh")
    # MLC/edgellm-agnostic label for cross-framework comparison, e.g. int4_awq/int8_sq/fp16.
    ap.add_argument("--quant", required=True)
    ap.add_argument("--quant-class", required=True, choices=["q4", "q8", "fp16"])
    ap.add_argument("--model-name", default="Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out-dir", default="/workspace/results/llm-benchmarks")
    args = ap.parse_args()

    stats = benchmark(args.engine_dir, args.n_predict, args.runs, args.warmup)

    engine_size_mb = sum(f.stat().st_size for f in Path(args.engine_dir).rglob("*") if f.is_file()) / 1e6

    result = {
        "model": args.model_name,
        "framework": "tensorrt_edgellm",
        "quantization": args.quant,
        "quant_class": args.quant_class,
        "model_size_mb": engine_size_mb,
        "peak_memory_mb": None,  # filled in by scripts/tegrastats_summary.py on the host
        "avg_power_w": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **stats,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tensorrt_edgellm_{args.quant}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
