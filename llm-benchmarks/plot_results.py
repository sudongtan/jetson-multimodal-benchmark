"""Plot a framework-comparison chart (tokens/sec by quant level) from results/llm-benchmarks/*.json.

Run on the host (matplotlib doesn't need the Jetson container):
    python3 llm-benchmarks/plot_results.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FRAMEWORK_LABEL = {"llama_cpp": "llama.cpp", "mlc": "MLC", "tensorrt_edgellm": "TensorRT-Edge-LLM"}
FRAMEWORK_COLOR = {"llama_cpp": "tab:blue", "mlc": "tab:orange", "tensorrt_edgellm": "tab:green"}
QUANT_ORDER = ["q4", "q8", "fp16"]


def load_results(results_dir: Path):
    results = []
    for path in sorted(results_dir.glob("*.json")):
        results.append(json.loads(path.read_text()))
    return results


def plot(results, out_path: Path):
    frameworks = sorted({r["framework"] for r in results})
    quant_classes = [q for q in QUANT_ORDER if any(r["quant_class"] == q for r in results)]

    x = np.arange(len(quant_classes))
    width = 0.8 / len(frameworks)

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, framework in enumerate(frameworks):
        by_quant = {r["quant_class"]: r for r in results if r["framework"] == framework}
        heights = [by_quant[q]["tokens_per_sec"] if q in by_quant else 0 for q in quant_classes]
        offsets = [by_quant[q]["quantization"] if q in by_quant else "" for q in quant_classes]
        bars = ax.bar(
            x + i * width, heights, width,
            label=FRAMEWORK_LABEL.get(framework, framework), color=FRAMEWORK_COLOR.get(framework, "gray"), zorder=3,
        )
        for bar, label, height in zip(bars, offsets, heights):
            if height:
                ax.annotate(label, (bar.get_x() + bar.get_width() / 2, height),
                            textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)

    ax.set_xticks(x + width * (len(frameworks) - 1) / 2)
    ax.set_xticklabels([q.upper() for q in quant_classes])
    ax.set_xlabel("Quantization level")
    ax.set_ylabel("Decode tokens/sec")
    framework_title = " vs. ".join(FRAMEWORK_LABEL.get(f, f) for f in frameworks)
    ax.set_title(f"{results[0]['model']}: {framework_title}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/llm-benchmarks")
    ap.add_argument("--out", default="docs/llm-framework-comparison.png")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)
    if not results:
        raise SystemExit(f"no *.json files found in {results_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot(results, out_path)
