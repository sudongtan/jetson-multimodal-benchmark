"""Plot a latency-breakdown chart (image preprocessing vs. text prefill vs. decode)
from results/vlm-benchmarks/*.json -- most public VLM benchmarks report only a single
end-to-end latency number, so separating these out is the point of this chart.

Structured as grouped bars per stage (not stacked-by-quant): with a single result
that's just one bar per stage, each individually legible -- a stacked single bar
would collapse to one column with no comparison to make. With multiple
models/quant levels (e.g. Gemma-3-4B vs. Qwen2.5-VL-3B, see notes/vlm-benchmarks.md)
it becomes a proper grouped-bar comparison at each stage, same shape as
llm-benchmarks/plot_results.py.

Run on the host (matplotlib doesn't need the Jetson container):
    python3 vlm-benchmarks/plot_results.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STAGES = ["preprocessing_ms", "prefill_ms", "decode_ms"]
STAGE_LABEL = {
    "preprocessing_ms": "Image\npreprocessing",
    "prefill_ms": "Text\nprefill",
    "decode_ms": "Decode\n(generation)",
}
# One color per config (quant level, or later framework+quant); stable order so a
# color always means the same config across re-plots, not by whatever order dict
# iteration happens to produce.
CONFIG_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def load_results(results_dir: Path):
    results = []
    for path in sorted(results_dir.glob("*.json")):
        results.append(json.loads(path.read_text()))
    return results


def plot(results, out_path: Path):
    # one "config" = one result file, labeled by model + quant tag so results from
    # different models (e.g. Gemma-3-4B vs. Qwen2.5-VL-3B) don't share an ambiguous
    # "Q4_K_M" label.
    models = sorted({r["model"] for r in results})
    configs = [(f"{r['model']} {r['quantization']}", r) for r in results]

    x = np.arange(len(STAGES))
    width = 0.8 / max(len(configs), 1)
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (label, r) in enumerate(configs):
        heights = [r.get(stage) or 0.0 for stage in STAGES]
        offset = x + i * width - (len(configs) - 1) * width / 2
        bars = ax.bar(offset, heights, width, label=label, color=CONFIG_COLORS[i % len(CONFIG_COLORS)], zorder=3)
        ax.bar_label(bars, fmt="%.0f ms", padding=3, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABEL[s] for s in STAGES])
    ax.set_ylabel("Latency (ms)")
    title_model = models[0] if len(models) == 1 else f"{len(models)} models"
    ax.set_title(f"{title_model}: VLM latency breakdown ({results[0]['framework']})")
    if len(configs) > 1:
        ax.legend()  # only needed once there's more than one config to tell apart
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.margins(y=0.15)  # headroom so bar_label text doesn't clip the top

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/vlm-benchmarks")
    ap.add_argument("--out", default="docs/vlm-latency-breakdown.png")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)
    if not results:
        raise SystemExit(f"no *.json files found in {results_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot(results, out_path)
