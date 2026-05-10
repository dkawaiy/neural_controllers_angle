#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME=.hf_cache
export HF_DATASETS_CACHE=.hf_cache/datasets
export MPLCONFIGDIR=.mpl_cache
export XDG_CACHE_HOME=.cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1

PYTHON=.venv-mps/bin/python
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23)

echo "Checking Apple GPU / MPS..."
"$PYTHON" experiments/check_mac_gpu.py

if ! "$PYTHON" - <<'PY'
import torch
raise SystemExit(0 if torch.backends.mps.is_available() else 1)
PY
then
  echo "MPS is not available to this process. Run this script from the normal macOS Terminal app."
  exit 1
fi

echo "Running base angle probe on MPS..."
"$PYTHON" experiments/translation_angle_probe.py \
  --device mps \
  --dtype float16 \
  --model-id "$MODEL_ID" \
  --model-state base_mps \
  --batch-size 4 \
  --layers "${LAYERS[@]}"
BASE_RUN=$(ls -td artifacts/translation_angle/* | head -1)

echo "Training translation LoRA on MPS..."
"$PYTHON" experiments/train_translation_lora.py \
  --device mps \
  --dtype float16 \
  --model-id "$MODEL_ID" \
  --batch-size 1 \
  --grad-accum 4 \
  --max-steps 20 \
  --learning-rate 5e-4 \
  --lora-r 4 \
  --lora-alpha 8 \
  --target-modules q_proj v_proj
LORA_RUN=$(ls -td artifacts/loras/translation_en_zh/* | head -1)

echo "Running LoRA angle probe on MPS..."
"$PYTHON" experiments/translation_angle_probe.py \
  --device mps \
  --dtype float16 \
  --model-id "$MODEL_ID" \
  --lora-path "$LORA_RUN" \
  --model-state translation_lora_mps \
  --batch-size 4 \
  --layers "${LAYERS[@]}"
LORA_ANGLE_RUN=$(ls -td artifacts/translation_angle/* | head -1)

echo "Comparing base vs LoRA..."
"$PYTHON" experiments/compare_translation_angles.py \
  --base-metrics "$BASE_RUN/metrics.json" \
  --lora-metrics "$LORA_ANGLE_RUN/metrics.json"
COMPARISON_RUN=$(ls -td artifacts/translation_angle_comparison/* | head -1)

echo
echo "Done."
echo "Base angle run:       $BASE_RUN"
echo "LoRA adapter:         $LORA_RUN"
echo "LoRA angle run:       $LORA_ANGLE_RUN"
echo "Comparison artifacts: $COMPARISON_RUN"
