#!/usr/bin/env bash
# Quantize (optional) + export to ONNX + build a TensorRT engine, per NVIDIA/TensorRT-Edge-LLM
# v0.5.0's own CLI (tensorrt-edgellm-quantize-llm / tensorrt-edgellm-export-llm / llm_build,
# all on PATH inside the tensorrt_edgellm jetson-container). Reused as-is, just parametrized.
# Run inside the tensorrt_edgellm jetson-container:
#   MODEL_PATH=/workspace/.cache/models/Qwen2.5-1.5B-Instruct QUANT=int4_awq \
#   WORKSPACE=/workspace/.cache/edgellm bash build_engine_edgellm.sh
set -euo pipefail

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the local unquantized HF model dir}
WORKSPACE=${WORKSPACE:?set WORKSPACE to a scratch dir for onnx/engine artifacts}
# "" (unset) = fp16 baseline, no quantization step. Otherwise one of: int4_awq, int8_sq
# (fp8/nvfp4/mxfp8 need Hopper+ tensor cores, not available on Orin's Ampere GPU).
QUANT=${QUANT:-}
TAG=${QUANT:-fp16}

ENGINE_DIR="$WORKSPACE/engines/$TAG"
if [ -f "$ENGINE_DIR/.done" ]; then
  echo "already built: $ENGINE_DIR"
  exit 0
fi

EXPORT_INPUT="$MODEL_PATH"

if [ -n "$QUANT" ]; then
  QUANT_DIR="$WORKSPACE/quantized/$TAG"
  tensorrt-edgellm-quantize-llm --model_dir "$MODEL_PATH" --output_dir "$QUANT_DIR" \
    --quantization "$QUANT" --dtype fp16
  EXPORT_INPUT="$QUANT_DIR"
fi

ONNX_DIR="$WORKSPACE/onnx/$TAG"
tensorrt-edgellm-export-llm --model_dir "$EXPORT_INPUT" --output_dir "$ONNX_DIR" --device cuda

llm_build --onnxDir "$ONNX_DIR/llm" --engineDir "$ENGINE_DIR" \
  --maxBatchSize 1 --maxInputLen 2048 --maxKVCacheCapacity 2048

touch "$ENGINE_DIR/.done"
echo "wrote $ENGINE_DIR"
