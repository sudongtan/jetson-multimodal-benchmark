#!/usr/bin/env bash
# Export FP32/FP16/INT8 engines for a YOLO model and benchmark each, inside
# the ultralytics jetson container. Run from the repo root:
#   bash vision-benchmarks/run_all.sh
set -euo pipefail

MODEL=${MODEL:-yolov8n.pt}
# Dataset config name (Ultralytics built-in, e.g. coco8.yaml/coco128.yaml) — used for
# INT8 calibration and for the mAP check in benchmark.py. Override with: DATA=coco128.yaml
DATA=${DATA:-coco128.yaml}
IMAGE=ultralytics/ultralytics:latest-jetson-jetpack6

# Ultralytics auto-downloads dataset images the first time a --data config is used.
# Mounting .cache/datasets (on the host, under this repo) to /datasets (where Ultralytics
# looks by default inside the container) means that download persists across container
# runs instead of re-fetching every time. Actual files end up at:
#   .cache/datasets/<name>/images/... and .cache/datasets/<name>/labels/...
# This folder is gitignored — it's a local cache, not something to commit.
mkdir -p .cache/datasets

docker run --rm -it --ipc=host --runtime=nvidia \
  -v "$(pwd)":/workspace -w /workspace/vision-benchmarks \
  -v "$(pwd)/.cache/datasets":/datasets \
  "$IMAGE" bash -c "
    set -e
    for p in fp32 fp16 int8; do
      echo \"=== export \$p ===\"
      python3 export.py --model $MODEL --precision \$p --data $DATA
    done
    for p in fp32 fp16 int8; do
      echo \"=== benchmark \$p ===\"
      python3 benchmark.py --engine ${MODEL%.pt}_\${p}.engine --precision \$p --data $DATA
    done
  "
