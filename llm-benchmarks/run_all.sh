#!/usr/bin/env bash
# Baseline LLM benchmark: Qwen2.5-1.5B-Instruct via llama.cpp (GGUF, Q4_K_M/Q8_0/FP16),
# MLC (q4f16_1/q0f16), and TensorRT-Edge-LLM (int4_awq/int8_sq/fp16), each wrapped in a
# tegrastats capture for peak RAM / avg power. Uses jetson-containers' own run.sh/autotag
# (per this repo's roadmap) rather than a pinned image tag, since these images are only
# published through its registry.
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
        MODEL_PATH=\"\$HF_MODEL_DIR\" QUANT='$quant' CONV_TEMPLATE=chatml bash quantize_mlc.sh
        OUT=/data/models/mlc/dist/${MODEL_NAME}-${quant}
        python3 benchmark_mlc.py --model \"\$OUT\" --model-lib \"\$OUT/${MODEL_NAME}-${quant}-cuda.so\" \
          --quant '$quant' --quant-class '$quant_class' --model-name '$MODEL_NAME'
      "
done

# TensorRT-Edge-LLM (NVIDIA/TensorRT-Edge-LLM v0.5.0, via nvidia_modelopt): int4_awq (~Q4),
# int8_sq (~Q8), and no quantization for the FP16 baseline -- all three run on Ampere,
# unlike its fp8/nvfp4/mxfp8 schemes which need Hopper+ tensor cores.
mkdir -p .cache/edgellm
# "unquantized" builds the fp16 baseline (build_engine_edgellm.sh names its engine dir
# "fp16" in that case, via QUANT="" -- $tag below mirrors that same fallback).
for quant_spec in "int4_awq q4" "int8_sq q8" "unquantized fp16"; do
  read -r quant_env quant_class <<< "$quant_spec"
  if [ "$quant_env" = "unquantized" ]; then quant_env=""; fi
  tag=${quant_env:-fp16}
  echo "=== tensorrt_edgellm $tag ==="
  run_with_tegrastats "results/llm-benchmarks/tensorrt_edgellm_${tag}.json" \
    docker run --rm --ipc=host --runtime=nvidia \
      -v "$(pwd)":/workspace -w /workspace/llm-benchmarks \
      "$("$JETSON_CONTAINERS/autotag" tensorrt_edgellm)" /bin/bash -c "
        set -e
        HF_MODEL_DIR=\$(huggingface-downloader --cache-dir /workspace/.cache/models '$HF_MODEL')
        MODEL_PATH=\"\$HF_MODEL_DIR\" QUANT='$quant_env' WORKSPACE=/workspace/.cache/edgellm bash build_engine_edgellm.sh
        python3 benchmark_tensorrt_edgellm.py --engine-dir /workspace/.cache/edgellm/engines/${tag} \
          --quant '$tag' --quant-class '$quant_class' --model-name '$MODEL_NAME'
      "
done

echo "done. results in results/llm-benchmarks/"
