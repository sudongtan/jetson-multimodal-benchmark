#!/usr/bin/env bash
# Phase 3 baseline: two VLMs via llama.cpp's multimodal CLI (both official ggml-org
# GGUF conversions, both confirmed to include a working mmproj file), reusing Phase 2's
# already-working llama_cpp container (confirmed to already support --mmproj/--image).
# Each run wrapped in tegrastats for peak RAM / avg power.
#
# Two models, not one, specifically to test whether "Q4-only" is a property of this
# board or of Gemma-3-4B's size: Gemma-3-4B (tested, Q4_K_M only -- Q8_0/f16 crash,
# see the per-model comment below) and Qwen2.5-VL-3B (smaller, untested -- Q4_K_M
# should have real headroom this time instead of Gemma's ~300MB margin, Q8_0 is a
# genuine open question rather than a predictable failure). See notes/vlm-benchmarks.md
# for the full reasoning and size math behind this comparison.
#
# A single quant/model combination crashing (e.g. a repeat of Gemma's Q8_0 SIGSEGV)
# does NOT abort the whole sweep -- each run is allowed to fail and move on, since part
# of the point here is finding where the failures actually are.
#
# Run from the repo root:
#   bash vlm-benchmarks/run_all.sh
set -uo pipefail  # deliberately not -e: a single config's docker run may legitimately fail

JETSON_CONTAINERS=${JETSON_CONTAINERS:-/mnt/jetson-data/jetson-containers}
# A stable, widely-used ML test image (COCO val2017, "two cats on a couch").
TEST_IMAGE_URL=${TEST_IMAGE_URL:-http://images.cocodataset.org/val2017/000000039769.jpg}

mkdir -p .cache/models .cache/images results/vlm-benchmarks

# One shared test image for every run, downloaded once on the host.
IMAGE_PATH=.cache/images/test.jpg
if [ ! -f "$IMAGE_PATH" ]; then
  curl -sL "$TEST_IMAGE_URL" -o "$IMAGE_PATH"
fi

# Captures tegrastats around $2.. and merges peak RAM / avg power into $1 (a result JSON
# that the wrapped command is expected to have just written).
run_with_tegrastats() {
  local result_json=$1; shift
  local log
  log=$(mktemp)
  tegrastats --interval 200 --logfile "$log" &
  local tegra_pid=$!
  "$@"
  local status=$?
  kill "$tegra_pid" 2>/dev/null || true
  wait "$tegra_pid" 2>/dev/null || true
  if [ "$status" -eq 0 ]; then
    python3 scripts/tegrastats_summary.py "$log" "$result_json"
  fi
  rm -f "$log"
  return "$status"
}

# Each entry: HF_REPO MODEL_NAME MMPROJ_FILE "quant1 quant2 ..."
run_model() {
  local hf_repo=$1 model_name=$2 mmproj_file=$3 quants=$4
  for quant_spec in $quants; do
    read -r quant quant_class <<< "${quant_spec//,/ }"
    echo "=== llama.cpp (VLM) $model_name $quant ==="
    if ! run_with_tegrastats "results/vlm-benchmarks/llama_cpp_${model_name}_${quant}.json" \
      docker run --rm --ipc=host --runtime=nvidia \
        -v "$(pwd)":/workspace -w /workspace/vlm-benchmarks \
        "$("$JETSON_CONTAINERS/autotag" llama_cpp)" /bin/bash -c "
          set -e
          MODEL=\$(huggingface-downloader --cache-dir /workspace/.cache/models '$hf_repo/${model_name}-${quant}.gguf')
          MMPROJ=\$(huggingface-downloader --cache-dir /workspace/.cache/models '$hf_repo/$mmproj_file')
          python3 benchmark_llama_cpp_vlm.py --model \"\$MODEL\" --mmproj \"\$MMPROJ\" \
            --image /workspace/$IMAGE_PATH --quant '$quant' --quant-class '$quant_class' --model-name '$model_name'
          chown -R \"\$(stat -c '%u:%g' /workspace)\" /workspace/results /workspace/.cache
        "; then
      echo "!!! $model_name $quant FAILED -- see output above, continuing with the next config"
    fi
  done
}

# Gemma-3-4B: Q4_K_M only -- Q8_0/f16 do NOT fit on this 8GB board and crash rather
# than fail cleanly. Measured: Q4_K_M + mmproj peaks at 7097MB RSS (of 7.4GB total),
# leaving ~300MB headroom. Q8_0 (4.13GB weights alone, vs. Q4_K_M's 2.49GB) exceeds
# what's left after the LLM loads, and hits a real llama.cpp bug when it does:
# clip.cpp's alloc_compute_meta ignores ggml_backend_sched_reserve's return value, so
# a failed vision-encoder buffer allocation under memory pressure segfaults instead of
# erroring cleanly -- see https://github.com/ggml-org/llama.cpp/issues/23422
# (confirmed architecture-independent, not specific to this model or board).
run_model ggml-org/gemma-3-4b-it-GGUF gemma-3-4b-it mmproj-model-f16.gguf "Q4_K_M,q4"

# Qwen2.5-VL-3B: smaller than Gemma-3-4B, so Q4_K_M should have real headroom this
# time. Q8_0 is left in deliberately -- per the size math in notes/vlm-benchmarks.md
# it's a genuine open question (may hit the same #23422 failure, may not), and that
# uncertainty is exactly what this second model is here to resolve. f16 (6.18GB
# weights alone) is skipped -- essentially certain to fail the same way Gemma's did,
# not worth the download time to confirm.
run_model ggml-org/Qwen2.5-VL-3B-Instruct-GGUF Qwen2.5-VL-3B-Instruct mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf "Q4_K_M,q4 Q8_0,q8"

echo "done. results in results/vlm-benchmarks/"
