#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# One-command CUDA server entrypoint.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 bash experiments/run_server_rigorous_translation_geometry.sh
#
# Fast smoke:
#   CUDA_VISIBLE_DEVICES=0 SEEDS="42" MAX_STEPS=5 BOOTSTRAP=20 bash experiments/run_server_rigorous_translation_geometry.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.hf_cache/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PWD/.hf_cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PWD/.hf_cache/transformers}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mpl_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
export TOKENIZERS_PARALLELISM=false

PYTHON="${PYTHON:-$PWD/.venv-server/bin/python}"
VENV_DIR="$(dirname "$(dirname "$PYTHON")")"

if [[ ! -x "$PYTHON" ]]; then
  echo "Creating server venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
  "$PYTHON" -m pip install --upgrade pip setuptools wheel
  "$PYTHON" -m pip install -r requirements-gpu-cu118.txt
fi

echo "Using Python: $PYTHON"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using HF_ENDPOINT=$HF_ENDPOINT"

"$PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Check nvidia-smi and container GPU access.")
print("device 0:", torch.cuda.get_device_name(0))
print("test tensor:", torch.ones(1, device="cuda:0"))
PY

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
SEEDS="${SEEDS:-42 43 44}"
MAX_STEPS="${MAX_STEPS:-60}"
BOOTSTRAP="${BOOTSTRAP:-1000}"
EXTRACTORS="${EXTRACTORS:-mean_diff logistic rfm}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
BOOTSTRAP_RFM="${BOOTSTRAP_RFM:-0}"
BOOTSTRAP_MLP_AGOP="${BOOTSTRAP_MLP_AGOP:-0}"
DATA_SOURCE="${DATA_SOURCE:-real}"
DATA_PATH="${DATA_PATH:-data/translation_real/en_zh_swaption20k_sample.jsonl}"

echo "Starting rigorous translation geometry run..."
echo "MODEL_ID=$MODEL_ID"
echo "DEVICE=$DEVICE"
echo "DTYPE=$DTYPE"
echo "SEEDS=$SEEDS"
echo "MAX_STEPS=$MAX_STEPS"
echo "BOOTSTRAP=$BOOTSTRAP"
echo "EXTRACTORS=$EXTRACTORS"
echo "BOOTSTRAP_RFM=$BOOTSTRAP_RFM"
echo "DATA_SOURCE=$DATA_SOURCE"
echo "DATA_PATH=$DATA_PATH"

PYTHON="$PYTHON" \
DEVICE="$DEVICE" \
DTYPE="$DTYPE" \
SEEDS="$SEEDS" \
MAX_STEPS="$MAX_STEPS" \
BOOTSTRAP="$BOOTSTRAP" \
EXTRACTORS="$EXTRACTORS" \
MODEL_ID="$MODEL_ID" \
BOOTSTRAP_RFM="$BOOTSTRAP_RFM" \
BOOTSTRAP_MLP_AGOP="$BOOTSTRAP_MLP_AGOP" \
DATA_SOURCE="$DATA_SOURCE" \
DATA_PATH="$DATA_PATH" \
bash experiments/run_rigorous_translation_geometry.sh
