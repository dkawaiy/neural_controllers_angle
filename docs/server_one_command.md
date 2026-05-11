# Server One-Command Run

Clone your branch on the GPU server/container, then run one command.

```bash
git clone -b knowledge-geometry-experiments https://github.com/dkawaiy/neural_controllers_angle.git
cd neural_controllers_angle
CUDA_VISIBLE_DEVICES=0 bash experiments/run_server_rigorous_translation_geometry.sh
```

The script will:

- create `.venv-server` if missing;
- install `requirements-gpu-cu118.txt`;
- set Hugging Face mirror via `HF_ENDPOINT=https://hf-mirror.com`;
- run CUDA sanity checks;
- generate synthetic lexicon splits;
- train real LoRA and random-target LoRA for seeds `42 43 44`;
- run geometry probes with `mean_diff`, `logistic`, and native `rfm`;
- bootstrap mean-diff/logistic test angles; RFM is run once per layer/seed by default because bootstrapping it is much slower;
- evaluate heldout synthetic-term behavior;
- write aggregate summaries under `artifacts/translation_rigorous_summary/`.

Fast smoke:

```bash
CUDA_VISIBLE_DEVICES=0 SEEDS="42" MAX_STEPS=5 BOOTSTRAP=20 \
  bash experiments/run_server_rigorous_translation_geometry.sh
```

Formal default:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/run_server_rigorous_translation_geometry.sh
```

Useful overrides:

```bash
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct CUDA_VISIBLE_DEVICES=0 \
  bash experiments/run_server_rigorous_translation_geometry.sh

EXTRACTORS="mean_diff logistic" CUDA_VISIBLE_DEVICES=0 \
  bash experiments/run_server_rigorous_translation_geometry.sh

EXTRACTORS="mean_diff logistic rfm" BOOTSTRAP_RFM=1 CUDA_VISIBLE_DEVICES=0 \
  bash experiments/run_server_rigorous_translation_geometry.sh
```

If CUDA fails before the experiment starts, fix GPU/container access first:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
print(torch.ones(1, device="cuda:0"))
PY
```
