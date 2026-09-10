#!/usr/bin/env bash
# Baseline LLM benchmark (the roadmap's actual Phase 2 requirement): Qwen2.5-1.5B-Instruct
# via llama.cpp (GGUF, Q4_K_M/Q8_0/FP16) and MLC (q4f16_1/q0f16), each wrapped in a
# tegrastats capture for peak RAM / avg power. Uses jetson-containers' own run.sh/autotag
# (per this repo's roadmap) rather than a pinned image tag, since these images are only
# published through its registry.
#
# TensorRT-Edge-LLM is intentionally NOT included here -- see
# notes/llm-benchmarks.md ("Is TensorRT actually a viable way to deploy an LLM on this
# board at all?") for why: no prebuilt image, no documented successful build on an Orin
# Nano by anyone, and NVIDIA's own TensorRT-LLM engineers redirect users away from the
# TensorRT engine workflow on this exact board. It's kept as an optional stretch goal in
# run_tensorrt_edgellm.sh, run separately, not as part of the baseline.
#
# Run from the repo root:
#   bash llm-benchmarks/run_all.sh
set -euo pipefail

JETSON_CONTAINERS=${JETSON_CONTAINERS:-/mnt/jetson-data/jetson-containers}
HF_MODEL=${HF_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
HF_GGUF_REPO=${HF_GGUF_REPO:-Qwen/Qwen2.5-1.5B-Instruct-GGUF}
MODEL_NAME=${MODEL_NAME:-Qwen2.5-1.5B-Instruct}

mkdir -p .cache/models .cache/mlc results/llm-benchmarks

# Captures tegrastats around $2.. and merges peak RAM / avg power into $1 (a result JSON
# that the wrapped command is expected to have just written).
run_with_tegrastats() {
  local result_json=$1; shift
  local log
  log=$(mktemp)
  tegrastats --interval 200 --logfile "$log" &
  local tegra_pid=$!
  "$@"
  kill "$tegra_pid" 2>/dev/null || true
  wait "$tegra_pid" 2>/dev/null || true
  python3 scripts/tegrastats_summary.py "$log" "$result_json"
  rm -f "$log"
}

# llama.cpp: three GGUF quant levels, one docker run each (tegrastats wraps the whole run).
for spec in \
  "qwen2.5-1.5b-instruct-q4_k_m.gguf q4_k_m q4" \
  "qwen2.5-1.5b-instruct-q8_0.gguf q8_0 q8" \
  "qwen2.5-1.5b-instruct-fp16.gguf fp16 fp16" \
; do
  read -r gguf_file quant quant_class <<< "$spec"
  echo "=== llama.cpp $quant ==="
  run_with_tegrastats "results/llm-benchmarks/llama_cpp_${quant}.json" \
    docker run --rm --ipc=host --runtime=nvidia \
      -v "$(pwd)":/workspace -w /workspace/llm-benchmarks \
      "$("$JETSON_CONTAINERS/autotag" llama_cpp)" /bin/bash -c "
        set -e
        MODEL=\$(huggingface-downloader --cache-dir /workspace/.cache/models '$HF_GGUF_REPO/$gguf_file')
        python3 benchmark_llama_cpp.py --model \"\$MODEL\" --quant '$quant' --quant-class '$quant_class' --model-name '$MODEL_NAME'
        chown -R \"\$(stat -c '%u:%g' /workspace)\" /workspace/results /workspace/.cache
      "
done

# MLC: no native int8 weight-only scheme is well-supported on Orin's Ampere GPU (its 8-bit
# paths are fp8, which needs Hopper-class tensor cores), so we sweep the two schemes that
# are: q4f16_1 (~Q4) and q0f16 (unquantized weights, fp16 -- the FP16 baseline).
for quant_spec in "q4f16_1 q4" "q0f16 fp16"; do
  read -r quant quant_class <<< "$quant_spec"
  echo "=== mlc $quant ==="
  run_with_tegrastats "results/llm-benchmarks/mlc_${quant}.json" \
    docker run --rm --ipc=host --runtime=nvidia \
      -v "$(pwd)":/workspace -w /workspace/llm-benchmarks \
      -v "$(pwd)/.cache/mlc":/data/models/mlc \
      "$("$JETSON_CONTAINERS/autotag" mlc)" /bin/bash -c "
        set -e
        HF_MODEL_DIR=\$(huggingface-downloader --cache-dir /workspace/.cache/models '$HF_MODEL')
        MODEL_PATH=\"\$HF_MODEL_DIR\" QUANT='$quant' CONV_TEMPLATE=chatml MODEL_NAME='$MODEL_NAME' bash quantize_mlc.sh
        OUT=/data/models/mlc/dist/${MODEL_NAME}-${quant}
        python3 benchmark_mlc.py --model \"\$OUT\" --model-lib \"\$OUT/${MODEL_NAME}-${quant}-cuda.so\" \
          --quant '$quant' --quant-class '$quant_class' --model-name '$MODEL_NAME'
        chown -R \"\$(stat -c '%u:%g' /workspace)\" /workspace/results /workspace/.cache
      "
done

echo "done. results in results/llm-benchmarks/"
echo "(TensorRT-Edge-LLM is a separate, optional stretch goal -- see run_tensorrt_edgellm.sh)"
