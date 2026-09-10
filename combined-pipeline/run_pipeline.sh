#!/usr/bin/env bash
# The Phase 4 capstone: camera in, language out. Runs the two stages in their
# own containers (vision inference needs ultralytics/TensorRT, description needs
# llama_cpp's mtmd support -- no single prebuilt image has both), handing off
# through .cache/ (bind-mounted, shared by both containers) rather than a live
# socket/API between them, matching this project's established one-shot,
# benchmarkable style (see llm-benchmarks/, vlm-benchmarks/) rather than a
# persistent server pair.
#
# Vision runs every loop iteration (cheap: ~8ms warmed up). The VLM stage only
# runs when the detected class actually changes from the previous iteration --
# NOT every frame. This isn't a shortcut, it's the actual pattern real edge VLM
# deployments use: NVIDIA's own Live VLM WebUI runs the VLM asynchronously in
# the background rather than per-frame (~7-8 sec/frame on this same board,
# matching what this project measured), and the broader edge-AI literature is
# explicit that continuous per-frame VLM inference isn't how this gets deployed
# in practice -- see notes/combined-pipeline.md for the sources. A fast detector
# running continuously plus a slow VLM firing only on meaningful change is the
# realistic architecture, not a workaround.
#
# Prerequisites:
#   - A webcam connected, visible as /dev/video0 (override with VIDEO_DEVICE=/dev/video1 etc.)
#   - vision-benchmarks/yolov8n_fp16.engine already built. No .engine files ship in the
#     repo (see .gitignore) -- but that's just export.py, not the full benchmark suite:
#       docker run --rm --ipc=host --runtime=nvidia \
#         -v "$(pwd)":/workspace -w /workspace/vision-benchmarks \
#         ultralytics/ultralytics:latest-jetson-jetpack6 \
#         python3 export.py --model yolov8n.pt --precision fp16
#     (vision-benchmarks/run_all.sh also produces this file, but only as a side effect
#     of running all three precisions through a 200-iteration benchmark + mAP
#     validation each -- overkill if you just need the engine.)
#
# Run from the repo root:
#   bash combined-pipeline/run_pipeline.sh
# Run continuously (e.g. for recording a live demo), re-capturing a fresh
# frame every LOOP_INTERVAL seconds:
#   LOOP_COUNT=20 LOOP_INTERVAL=3 bash combined-pipeline/run_pipeline.sh
set -uo pipefail  # not -e: one bad frame/detection shouldn't kill a running demo loop

JETSON_CONTAINERS=${JETSON_CONTAINERS:-/mnt/jetson-data/jetson-containers}
VISION_IMAGE=ultralytics/ultralytics:latest-jetson-jetpack6
VIDEO_DEVICE=${VIDEO_DEVICE:-/dev/video0}
ENGINE=${ENGINE:-vision-benchmarks/yolov8n_fp16.engine}
HF_REPO=${HF_REPO:-ggml-org/gemma-3-4b-it-GGUF}
MODEL_NAME=${MODEL_NAME:-gemma-3-4b-it}
QUANT=${QUANT:-Q4_K_M}   # Q4_K_M only -- see notes/vlm-benchmarks.md for why Q8_0/f16 crash on this board
MMPROJ_FILE=${MMPROJ_FILE:-mmproj-model-f16.gguf}
LOOP_COUNT=${LOOP_COUNT:-1}
LOOP_INTERVAL=${LOOP_INTERVAL:-0}
# Decode time scales ~linearly with token count -- 128 (a full paragraph) costs
# ~8s alone at this model's ~15-18 tok/s. A short live-caption use case doesn't
# need that; try N_PREDICT=30 for a one-sentence answer instead.
N_PREDICT=${N_PREDICT:-128}

if [ ! -f "$ENGINE" ]; then
  echo "error: $ENGINE not found -- build it first (see the export.py command in this script's header comment, not the full vision-benchmarks/run_all.sh suite)" >&2
  exit 1
fi

# cv2.VideoCapture takes an integer index (V4L2 convention: /dev/video0 -> 0), while
# --device takes the device node path -- both must point at the same webcam.
VIDEO_INDEX=$(echo "$VIDEO_DEVICE" | grep -o '[0-9]*$')

mkdir -p combined-pipeline/.cache .cache/models results/combined-pipeline

last_class=""  # sentinel: no detection has been described yet this run

for i in $(seq 1 "$LOOP_COUNT"); do
  echo "=== run $i/$LOOP_COUNT ==="
  t_pipeline_start=$(date +%s.%N)

  echo "--- stage 1: capture + detect ---"
  docker run --rm --ipc=host --runtime=nvidia --device="$VIDEO_DEVICE" \
    -v "$(pwd)":/workspace -w /workspace/vision-benchmarks \
    "$VISION_IMAGE" python3 ../combined-pipeline/capture_and_detect.py --engine "$(basename "$ENGINE")" --video-device "$VIDEO_INDEX"
  stage1_status=$?

  if [ "$stage1_status" -ne 0 ]; then
    echo "!!! stage 1 (capture/detect) failed, skipping stage 2 for this run"
    sleep "$LOOP_INTERVAL"
    continue
  fi

  current_class=$(python3 -c "
import json
d = json.load(open('combined-pipeline/.cache/detection.json'))
print(d['detection']['class_name'] if d['detection'] else '(none)')
")

  if [ "$current_class" == "$last_class" ]; then
    echo "--- no change ($current_class), skipping VLM description ---"
    sleep "$LOOP_INTERVAL"
    continue
  fi
  last_class="$current_class"

  echo "--- stage 2: describe (detected class changed to: $current_class) ---"
  docker run --rm --ipc=host --runtime=nvidia \
    -v "$(pwd)":/workspace -w /workspace/combined-pipeline \
    "$("$JETSON_CONTAINERS/autotag" llama_cpp)" /bin/bash -c "
      set -e
      MODEL=\$(huggingface-downloader --cache-dir /workspace/.cache/models '$HF_REPO/${MODEL_NAME}-${QUANT}.gguf')
      MMPROJ=\$(huggingface-downloader --cache-dir /workspace/.cache/models '$HF_REPO/$MMPROJ_FILE')
      python3 describe_crop.py --model \"\$MODEL\" --mmproj \"\$MMPROJ\" --n-predict $N_PREDICT
      chown -R \"\$(stat -c '%u:%g' /workspace)\" /workspace/results /workspace/.cache /workspace/combined-pipeline/.cache
    "
  stage2_status=$?

  t_pipeline_end=$(date +%s.%N)

  if [ "$stage2_status" -eq 0 ] && [ -f combined-pipeline/.cache/result.json ]; then
    ts=$(date +%Y%m%dT%H%M%S)
    python3 -c "
import json, sys
result = json.load(open('combined-pipeline/.cache/result.json'))
result['wall_clock_ms'] = ($t_pipeline_end - $t_pipeline_start) * 1000  # includes docker startup/teardown, unlike end_to_end_ms
out_path = 'results/combined-pipeline/run_${ts}.json'
json.dump(result, open(out_path, 'w'), indent=2)
print(f'wrote {out_path}')
print()
print('Detected:', result['detection']['class_name'] if result['detection'] else '(no detection, described full frame)')
print('Description:', result['description'])
print(f\"End-to-end, cold (capture->text, incl. one-time model loads): {result['end_to_end_ms']:.0f} ms\")
print(f\"  of which vision load {result['vision_load_ms']:.0f} ms, VLM load {result['vlm_load_ms']:.0f} ms\")
print(f\"Steady-state compute only (excl. capture + both models' load time): {result['compute_only_ms']:.0f} ms\")
print(f\"  of which time-to-first-token (vision + VLM preprocessing + prefill): {result['vision_inference_ms'] + result['ttft_ms']:.0f} ms\")
print(f\"  -- not measured via streaming (script waits for the full response either way), but an accurate number: the first token is ready right as prefill finishes\")
"
  else
    echo "!!! stage 2 (describe) failed"
  fi

  sleep "$LOOP_INTERVAL"
done

echo "done. results in results/combined-pipeline/, last frame+crop in combined-pipeline/.cache/"
