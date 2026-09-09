"""Measure TTFT and decode tokens/sec for an MLC-compiled model, log as JSON.

Uses MLCEngine's OpenAI-style chat.completions.create(stream_options={"include_usage": True}),
which reports prefill/decode token rates directly (see quantize_mlc.sh for the convert/compile
step this expects as input). Run inside the mlc jetson-container:
    python3 benchmark_mlc.py \\
        --model /data/models/mlc/dist/Qwen2.5-1.5B-Instruct-q4f16_1 \\
        --model-lib /data/models/mlc/dist/Qwen2.5-1.5B-Instruct-q4f16_1/Qwen2.5-1.5B-Instruct-q4f16_1-cuda.so \\
        --quant q4f16_1 --quant-class q4
"""
import argparse
import json
import time
from pathlib import Path

from mlc_llm import MLCEngine
from mlc_llm.serve import EngineConfig


def dir_size_mb(path: str) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def benchmark(model_path: str, model_lib: str, prompt: str, n_predict: int, runs: int, warmup: int):
    engine = MLCEngine(
        model_path,
        model_lib=model_lib,
        mode="interactive",
        engine_config=EngineConfig(max_num_sequence=1),
    )

    ttft_samples_ms = []
    decode_tokens_per_sec_samples = []
    input_tokens = None

    for run in range(warmup + runs):
        usage = None
        for response in engine.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_path,
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=n_predict,
        ):
            if response.usage is not None:
                usage = response.usage.extra

        if run < warmup:
            continue

        input_tokens = usage["prefill_tokens"]
        # TTFT ~= prefill time: for a single, non-batched request the whole prompt is
        # processed before the first output token can be produced.
        prefill_time_s = usage["prefill_tokens"] / usage["prefill_tokens_per_s"]
        ttft_samples_ms.append(prefill_time_s * 1000)
        decode_tokens_per_sec_samples.append(usage.get("decode_tokens_per_s", 0.0))

    del engine  # MLC hangs on exit unless the engine is released first

    return {
        "ttft_ms": sum(ttft_samples_ms) / len(ttft_samples_ms),
        "tokens_per_sec": sum(decode_tokens_per_sec_samples) / len(decode_tokens_per_sec_samples),
        "n_predict": n_predict,
        "input_tokens": input_tokens,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the mlc_llm-compiled model dir")
    ap.add_argument("--model-lib", required=True, help="path to the compiled -cuda.so model lib")
    # MLC's own quantization scheme name, e.g. q4f16_1/q0f16 (see quantize_mlc.sh).
    ap.add_argument("--quant", required=True)
    # Coarse bucket for cross-framework comparison (llama.cpp and MLC name quant levels differently).
    ap.add_argument("--quant-class", required=True, choices=["q4", "q8", "fp16"])
    ap.add_argument("--model-name", default="Qwen2.5-1.5B-Instruct")
    ap.add_argument("--prompt", default="Explain what a transformer neural network is, in simple terms.")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out-dir", default="/workspace/results/llm-benchmarks")
    args = ap.parse_args()

    stats = benchmark(args.model, args.model_lib, args.prompt, args.n_predict, args.runs, args.warmup)

    result = {
        "model": args.model_name,
        "framework": "mlc",
        "quantization": args.quant,
        "quant_class": args.quant_class,
        "model_size_mb": dir_size_mb(args.model),
        "peak_memory_mb": None,  # filled in by scripts/tegrastats_summary.py on the host
        "avg_power_w": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **stats,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mlc_{args.quant}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
