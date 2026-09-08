"""Plot accuracy (mAP) vs. speed (FPS) from results/vision-benchmarks/*.json.

Run on the host (matplotlib doesn't need the Jetson container):
    python3 vision-benchmarks/plot_results.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

PRECISION_COLOR = {"fp32": "tab:blue", "fp16": "tab:orange", "int8": "tab:green"}


def load_results(results_dir: Path):
    results = []
    for path in sorted(results_dir.glob("*.json")):
        results.append(json.loads(path.read_text()))
    return results


def plot(results, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))

    for r in results:
        precision = r["precision"]
        ax.scatter(
            r["fps"], r["map50_95"],
            s=100, color=PRECISION_COLOR.get(precision, "gray"), zorder=3,
        )
        label = precision.upper()
        if r.get("eval_set"):
            label += f" ({r['eval_set']})"
        ax.annotate(
            label,
            (r["fps"], r["map50_95"]),
            textcoords="offset points", xytext=(8, 6),
        )

    ax.set_xlabel("Speed (FPS)")
    ax.set_ylabel("Accuracy (mAP50-95)")
    ax.set_title("YOLOv8n: Accuracy vs. Speed by TensorRT Precision")
    ax.grid(True, alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/vision-benchmarks")
    ap.add_argument("--out", default="docs/vision-accuracy-vs-speed.png")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)
    if not results:
        raise SystemExit(f"no *.json files found in {results_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot(results, out_path)
