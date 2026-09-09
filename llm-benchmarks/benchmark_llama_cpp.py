"""Measure TTFT and decode tokens/sec for a GGUF model via llama-cpp-python, log as JSON.

Run inside the llama_cpp jetson-container:
    python3 benchmark_llama_cpp.py \\
        --model $(huggingface-downloader Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf) \\
        --quant q4_k_m --quant-class q4
"""
import argparse
import json
import time
from pathlib import Path

from llama_cpp import Llama


def benchmark(model_path: str, prompt: str, n_predict: int, ctx_size: int, runs: int, warmup: int):
    llm = Llama(model_path=model_path, n_ctx=ctx_size, n_gpu_layers=-1, verbose=False)
    input_tokens = llm.tokenize(prompt.encode("utf-8"))

    ttft_samples_ms = []
    decode_tokens_per_sec_samples = []

    for run in range(warmup + runs):
        llm.reset()
        first_token_time = None
        n_generated = 0

        t_start = time.perf_counter()
        for _ in llm.generate(input_tokens, top_k=40, top_p=0.95):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            n_generated += 1
            if n_generated >= n_predict:
                break
        t_end = time.perf_counter()

        if run < warmup:
            continue

        ttft_ms = (first_token_time - t_start) * 1000
        # steady-state decode rate: tokens after the first, over the time after TTFT
        # (the first token's latency is dominated by prompt processing, not decoding)
        decode_time = t_end - first_token_time
        decode_tokens_per_sec = (n_generated - 1) / decode_time if decode_time > 0 else 0.0

        ttft_samples_ms.append(ttft_ms)
        decode_tokens_per_sec_samples.append(decode_tokens_per_sec)

    return {
        "ttft_ms": sum(ttft_samples_ms) / len(ttft_samples_ms),
        "tokens_per_sec": sum(decode_tokens_per_sec_samples) / len(decode_tokens_per_sec_samples),
        "n_predict": n_predict,
        "input_tokens": len(input_tokens),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Path to the .gguf file (typically from `huggingface-downloader`, see docstring above).
    ap.add_argument("--model", required=True)
    # Label recorded in the output JSON for the quant used, e.g. q4_k_m/q8_0/fp16 (llama.cpp's own tag).
    ap.add_argument("--quant", required=True)
    # Coarse bucket for cross-framework comparison (llama.cpp and MLC name quant levels differently).
    ap.add_argument("--quant-class", required=True, choices=["q4", "q8", "fp16"])
    ap.add_argument("--model-name", default="Qwen2.5-1.5B-Instruct")
    ap.add_argument("--prompt", default="Explain what a transformer neural network is, in simple terms.")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--ctx-size", type=int, default=2048)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out-dir", default="/workspace/results/llm-benchmarks")
    args = ap.parse_args()

    stats = benchmark(args.model, args.prompt, args.n_predict, args.ctx_size, args.runs, args.warmup)

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"llama_cpp_{args.quant}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
