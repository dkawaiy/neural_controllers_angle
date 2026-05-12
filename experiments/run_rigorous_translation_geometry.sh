#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME=.hf_cache
export HF_DATASETS_CACHE=.hf_cache/datasets
export MPLCONFIGDIR=.mpl_cache
export XDG_CACHE_HOME=.cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1

PYTHON=${PYTHON:-.venv-mps/bin/python}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
DEVICE=${DEVICE:-mps}
DTYPE=${DTYPE:-float16}
MAX_STEPS=${MAX_STEPS:-60}
BOOTSTRAP=${BOOTSTRAP:-1000}
EXTRACTORS_STR=${EXTRACTORS:-mean_diff logistic rfm}
SEEDS_STR=${SEEDS:-42 43 44}
DATA_SOURCE=${DATA_SOURCE:-real}
if [[ "$DATA_SOURCE" == "synthetic" ]]; then
  DATA_PATH=${DATA_PATH:-data/translation_synthetic/synthetic_lexicon_en_zh.jsonl}
else
  DATA_PATH=${DATA_PATH:-data/translation_real/en_zh_swaption20k_sample.jsonl}
fi
LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23)
LATE_LAYERS=(17 18 19 20 21 22 23)

read -r -a SEEDS_ARRAY <<< "$SEEDS_STR"
read -r -a EXTRACTORS_ARRAY <<< "$EXTRACTORS_STR"
BOOTSTRAP_EXTRA_FLAGS=()
if [[ "${BOOTSTRAP_RFM:-0}" == "1" ]]; then
  BOOTSTRAP_EXTRA_FLAGS+=(--bootstrap-rfm)
fi
if [[ "${BOOTSTRAP_MLP_AGOP:-0}" == "1" ]]; then
  BOOTSTRAP_EXTRA_FLAGS+=(--bootstrap-mlp-agop)
fi

if [[ "$DATA_SOURCE" == "synthetic" ]]; then
  echo "Generating synthetic split data..."
  "$PYTHON" experiments/generate_synthetic_translation_data.py \
    --out-path "$DATA_PATH" \
    --seed 42 \
    --n-finetune "${N_FINETUNE:-96}" \
    --n-geometry-train "${N_GEOMETRY_TRAIN:-48}" \
    --n-geometry-test "${N_GEOMETRY_TEST:-48}" \
    --n-behavior-test "${N_BEHAVIOR_TEST:-24}"
elif [[ -f "$DATA_PATH" ]]; then
  echo "Using prepared real translation data: $DATA_PATH"
else
  echo "Missing prepared translation data: $DATA_PATH" >&2
  echo "On an internet-enabled login node, run:" >&2
  echo "  $PYTHON experiments/prepare_translation_dataset.py --out-path $DATA_PATH" >&2
  exit 1
fi

if [[ "$DEVICE" == "mps" ]]; then
  echo "Checking Apple GPU / MPS..."
  "$PYTHON" experiments/check_mac_gpu.py
fi

REAL_COMPARISONS=()
RANDOM_COMPARISONS=()
REAL_BEHAVIOR=()
RANDOM_BEHAVIOR=()

for SEED in "${SEEDS_ARRAY[@]}"; do
  echo
  echo "==== Seed $SEED: base angle ===="
  "$PYTHON" experiments/translation_angle_probe.py \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --model-id "$MODEL_ID" \
    --model-state "base_seed_${SEED}" \
    --data-path "$DATA_PATH" \
    --fit-split translation_geometry_train \
    --test-split translation_geometry_test \
    --extractors "${EXTRACTORS_ARRAY[@]}" \
    --bootstrap-samples "$BOOTSTRAP" \
    --bootstrap-seed "$SEED" \
    "${BOOTSTRAP_EXTRA_FLAGS[@]}" \
    --batch-size 4 \
    --layers "${LAYERS[@]}"
  BASE_RUN=$(ls -td artifacts/translation_angle/* | head -1)

  echo
  echo "==== Seed $SEED: real LoRA ===="
  "$PYTHON" experiments/train_translation_lora.py \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --model-id "$MODEL_ID" \
    --data-path "$DATA_PATH" \
    --train-splits translation_finetune_train \
    --control-type real \
    --seed "$SEED" \
    --batch-size 1 \
    --grad-accum 4 \
    --max-steps "$MAX_STEPS" \
    --learning-rate 5e-4 \
    --lora-r 4 \
    --lora-alpha 8 \
    --target-modules q_proj v_proj
  REAL_LORA=$(ls -td artifacts/loras/translation_en_zh/* | head -1)

  "$PYTHON" experiments/translation_angle_probe.py \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --model-id "$MODEL_ID" \
    --lora-path "$REAL_LORA" \
    --model-state "real_lora_seed_${SEED}" \
    --data-path "$DATA_PATH" \
    --fit-split translation_geometry_train \
    --test-split translation_geometry_test \
    --extractors "${EXTRACTORS_ARRAY[@]}" \
    --bootstrap-samples "$BOOTSTRAP" \
    --bootstrap-seed "$SEED" \
    "${BOOTSTRAP_EXTRA_FLAGS[@]}" \
    --batch-size 4 \
    --layers "${LAYERS[@]}"
  REAL_ANGLE=$(ls -td artifacts/translation_angle/* | head -1)

  "$PYTHON" experiments/evaluate_translation_behavior.py \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --model-id "$MODEL_ID" \
    --lora-path "$REAL_LORA" \
    --model-state "real_lora_seed_${SEED}" \
    --data-path "$DATA_PATH" \
    --split translation_behavior_test
  REAL_BEHAVIOR_RUN=$(ls -td artifacts/translation_behavior/* | head -1)
  REAL_BEHAVIOR+=("$REAL_BEHAVIOR_RUN/behavior_summary.json")

  "$PYTHON" experiments/compare_translation_angles.py \
    --base-metrics "$BASE_RUN/metrics.json" \
    --lora-metrics "$REAL_ANGLE/metrics.json" \
    --late-layers "${LATE_LAYERS[@]}"
  REAL_COMPARISON=$(ls -td artifacts/translation_angle_comparison/* | head -1)
  REAL_COMPARISONS+=("$REAL_COMPARISON/comparison_summary.json")

  echo
  echo "==== Seed $SEED: random-target LoRA ===="
  "$PYTHON" experiments/train_translation_lora.py \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --model-id "$MODEL_ID" \
    --data-path "$DATA_PATH" \
    --train-splits translation_finetune_train \
    --control-type random_target \
    --random-targets \
    --seed "$SEED" \
    --batch-size 1 \
    --grad-accum 4 \
    --max-steps "$MAX_STEPS" \
    --learning-rate 5e-4 \
    --lora-r 4 \
    --lora-alpha 8 \
    --target-modules q_proj v_proj
  RANDOM_LORA=$(ls -td artifacts/loras/translation_en_zh/* | head -1)

  "$PYTHON" experiments/translation_angle_probe.py \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --model-id "$MODEL_ID" \
    --lora-path "$RANDOM_LORA" \
    --model-state "random_target_lora_seed_${SEED}" \
    --data-path "$DATA_PATH" \
    --fit-split translation_geometry_train \
    --test-split translation_geometry_test \
    --extractors "${EXTRACTORS_ARRAY[@]}" \
    --bootstrap-samples "$BOOTSTRAP" \
    --bootstrap-seed "$SEED" \
    "${BOOTSTRAP_EXTRA_FLAGS[@]}" \
    --batch-size 4 \
    --layers "${LAYERS[@]}"
  RANDOM_ANGLE=$(ls -td artifacts/translation_angle/* | head -1)

  "$PYTHON" experiments/evaluate_translation_behavior.py \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --model-id "$MODEL_ID" \
    --lora-path "$RANDOM_LORA" \
    --model-state "random_target_lora_seed_${SEED}" \
    --data-path "$DATA_PATH" \
    --split translation_behavior_test
  RANDOM_BEHAVIOR_RUN=$(ls -td artifacts/translation_behavior/* | head -1)
  RANDOM_BEHAVIOR+=("$RANDOM_BEHAVIOR_RUN/behavior_summary.json")

  "$PYTHON" experiments/compare_translation_angles.py \
    --base-metrics "$BASE_RUN/metrics.json" \
    --lora-metrics "$RANDOM_ANGLE/metrics.json" \
    --late-layers "${LATE_LAYERS[@]}"
  RANDOM_COMPARISON=$(ls -td artifacts/translation_angle_comparison/* | head -1)
  RANDOM_COMPARISONS+=("$RANDOM_COMPARISON/comparison_summary.json")
done

echo
echo "==== Aggregate summary ===="
"$PYTHON" experiments/summarize_translation_controls.py \
  --real-comparisons "${REAL_COMPARISONS[@]}" \
  --random-comparisons "${RANDOM_COMPARISONS[@]}" \
  --real-behavior "${REAL_BEHAVIOR[@]}" \
  --random-behavior "${RANDOM_BEHAVIOR[@]}" \
  --late-layers "${LATE_LAYERS[@]}"

SUMMARY_RUN=$(ls -td artifacts/translation_rigorous_summary/* | head -1)
echo
echo "Done."
echo "Rigorous summary: $SUMMARY_RUN"
