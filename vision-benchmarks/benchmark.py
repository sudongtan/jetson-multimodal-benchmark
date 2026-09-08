"""Measure FPS/latency and mAP for a YOLO engine, log the result as JSON.

Run inside the ultralytics/ultralytics:latest-jetson-jetpack6 container:
    python3 benchmark.py --engine yolov8n_fp16.engine --precision fp16 --data coco8.yaml
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def percentile(values, p):
    """p50 = median latency, p95 = worst-case-ish latency (95% of runs were faster than this)."""
    return float(np.percentile(values, p))


def benchmark(engine_path: str, precision: str, data: str, imgsz: int, iters: int, warmup: int):
    # task="detect" tells Ultralytics how to interpret the engine's raw output tensors
    # (bounding boxes + class scores) since a .engine file has no task metadata attached.
    model = YOLO(engine_path, task="detect")

    # A random noise image, reused for every timing iteration. Fine for latency/FPS
    # (TensorRT's runtime cost doesn't depend on pixel content) but useless for mAP,
    # which is why accuracy is measured separately below via model.val() on real images.
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup: the first few inferences pay one-time costs (CUDA context init, memory
    # allocation, GPU clocks ramping up) that would skew the timed average if counted.
    for _ in range(warmup):
        model.predict(dummy, imgsz=imgsz, verbose=False, device=0)

    latencies_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=imgsz, verbose=False, device=0)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    # Accuracy check: runs the engine against a real labeled dataset and computes mAP
    # (mean Average Precision) against ground-truth boxes — this is what tells you how
    # much accuracy each precision level (fp32/fp16/int8) actually costs you.
    metrics = model.val(data=data, imgsz=imgsz, device=0, verbose=False)

    result = {
        "model": Path(engine_path).stem,
        "precision": precision,
        "eval_set": Path(data).stem,
        "engine_path": str(engine_path),
        "model_size_mb": Path(engine_path).stat().st_size / 1e6,
        "imgsz": imgsz,
        "iters": iters,
        "fps": 1000.0 / float(np.mean(latencies_ms)),
        "latency_ms": {
            "mean": float(np.mean(latencies_ms)),
            "p50": percentile(latencies_ms, 50),
            "p95": percentile(latencies_ms, 95),
        },
        "map50": float(metrics.box.map50),      # mAP at IoU threshold 0.5 (looser, higher-valued metric)
        "map50_95": float(metrics.box.map),     # mAP averaged over IoU 0.5-0.95 (COCO's standard, stricter metric)
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Path to the .engine file to benchmark, e.g. yolov8n_fp16.engine (produced by export.py).
    ap.add_argument("--engine", required=True)
    # Just a label recorded in the output JSON so you know which precision this run was —
    # it doesn't change how the engine runs (that was baked in at export time).
    ap.add_argument("--precision", required=True, choices=["fp32", "fp16", "int8"])
    # Dataset yaml used for the mAP check (model.val) — e.g. Ultralytics' built-in coco8.yaml
    # (8 images, fast smoke test) or coco128.yaml / full coco.yaml for a credible accuracy number.
    ap.add_argument("--data", default="coco8.yaml")
    # Input resolution the engine expects (must match what it was exported/built with).
    ap.add_argument("--imgsz", type=int, default=640)
    # How many timed inference calls to average latency/FPS over.
    ap.add_argument("--iters", type=int, default=200)
    # How many untimed inference calls to run first, to let GPU/CUDA warm up (see comment above).
    ap.add_argument("--warmup", type=int, default=20)
    # Where to write the result JSON — defaults to the path this repo's containers mount
    # the project root at (/workspace), matching run_all.sh's -v "$(pwd)":/workspace.
    ap.add_argument("--out-dir", default="/workspace/results/vision-benchmarks")
    args = ap.parse_args()

    result = benchmark(args.engine, args.precision, args.data, args.imgsz, args.iters, args.warmup)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result['model']}_{result['eval_set']}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
