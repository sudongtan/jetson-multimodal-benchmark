#!/usr/bin/env bash
# Convert + quantize + compile a HF model for MLC, per jetson-containers' own
# packages/llm/mlc/test.sh (quantize() function) -- reused as-is, just parametrized.
# Run inside the mlc jetson-container:
#   MODEL_PATH=/data/models/hf/Qwen2.5-1.5B-Instruct QUANT=q4f16_1 bash quantize_mlc.sh
set -euo pipefail

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the local unquantized HF model dir}
QUANT=${QUANT:?set QUANT to an mlc_llm quantization scheme, e.g. q4f16_1 / q0f16}
MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT=${MLC_DIST:-/data/models/mlc/dist}/${MODEL_NAME}-${QUANT}
CONV_TEMPLATE=${CONV_TEMPLATE:-chatml}
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-2048}

if [ -f "$OUTPUT/$MODEL_NAME-$QUANT-cuda.so" ]; then
  echo "already compiled: $OUTPUT/$MODEL_NAME-$QUANT-cuda.so"
  exit 0
fi

python3 -m mlc_llm convert_weight "$MODEL_PATH" --quantization "$QUANT" --output "$OUTPUT"
python3 -m mlc_llm gen_config "$MODEL_PATH" --quantization "$QUANT" --conv-template "$CONV_TEMPLATE" \
  --context-window-size "$MAX_CONTEXT_LEN" --max-batch-size 1 --output "$OUTPUT"
python3 -m mlc_llm compile "$OUTPUT" --device cuda --opt O3 --output "$OUTPUT/$MODEL_NAME-$QUANT-cuda.so"

echo "wrote $OUTPUT/$MODEL_NAME-$QUANT-cuda.so"
