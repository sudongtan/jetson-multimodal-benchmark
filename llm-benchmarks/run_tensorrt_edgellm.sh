#!/usr/bin/env bash
# OPTIONAL stretch goal, not part of the Phase 2 baseline (see run_all.sh).
#
# TensorRT-Edge-LLM (NVIDIA/TensorRT-Edge-LLM v0.5.0, via nvidia_modelopt): int4_awq (~Q4),
# int8_sq (~Q8), and no quantization for the FP16 baseline -- all three run on Ampere,
# unlike its fp8/nvfp4/mxfp8 schemes which need Hopper+ tensor cores.
#
# Before running this: read notes/llm-benchmarks.md in full, especially "Is TensorRT
# actually a viable way to deploy an LLM on this board at all?". Summary of why this is
# split out from the baseline: there is no prebuilt image (full from-source build of a
# 31-package chain), no documented case of anyone completing that build on an Orin Nano
# 8GB specifically, and NVIDIA's own TensorRT-LLM engineers redirect users to the PyTorch
# backend instead of the TensorRT engine workflow on this exact board (its AOT engine
# tuning phase can't fall back to swap when it runs out of memory). The jetson-containers
# build itself needs three fixes just to get partway through its dependency chain
# (--resource cpuset-cpus/memory, Triton's MAX_JOBS, torchao's TORCH_CUDA_ARCH_LIST +
# stripping its hardcoded SM90a cutlass extension) before even reaching this package --
# see notes/llm-benchmarks.md for the full debugging trail and exact patches.
#
# Run from the repo root, once the tensorrt_edgellm image builds successfully:
#   bash llm-benchmarks/run_tensorrt_edgellm.sh
set -euo pipefail

JETSON_CONTAINERS=${JETSON_CONTAINERS:-/mnt/jetson-data/jetson-containers}
HF_MODEL=${HF_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
MODEL_NAME=${MODEL_NAME:-Qwen2.5-1.5B-Instruct}

mkdir -p .cache/models .cache/edgellm results/llm-benchmarks

# Captures tegrastats around $2.. and merges peak RAM / avg power into $1 (a result JSON
# that the wrapped command is expected to have just written). Duplicated from run_all.sh
# rather than sourced, so this script stays runnable standalone.
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
        chown -R \"\$(stat -c '%u:%g' /workspace)\" /workspace/results /workspace/.cache
      "
done

echo "done. results in results/llm-benchmarks/"
