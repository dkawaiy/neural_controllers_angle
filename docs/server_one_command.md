# Server One-Command Run

Clone your branch on the GPU server/container, prepare the real translation sample on an internet-enabled login node, then run the GPU job.

```bash
git clone -b knowledge-geometry-experiments https://github.com/dkawaiy/neural_controllers_angle.git
cd neural_controllers_angle
python experiments/prepare_translation_dataset.py --out-path data/translation_real/en_zh_swaption20k_sample.jsonl
```

The default real-data sample uses `swaption2009/20k-en-zh-translation-pinyin-hsk` and writes:

```text
translation_finetune_train: 1000
translation_geometry_train: 500
translation_geometry_test: 500
translation_behavior_test: 200
```

Then submit/run on a GPU node:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/run_server_rigorous_translation_geometry.sh
```

The script will:

- create `.venv-server` if missing;
- install `requirements-gpu-cu118.txt`;
- set Hugging Face mirror via `HF_ENDPOINT=https://hf-mirror.com`;
- run CUDA sanity checks;
- read prepared real EN-ZH translation splits from `data/translation_real/en_zh_swaption20k_sample.jsonl`;
- train real LoRA and random-target LoRA for seeds `42 43 44`;
- run geometry probes with `mean_diff`, `logistic`, and native `rfm`;
- bootstrap mean-diff/logistic test angles; RFM is run once per layer/seed by default because bootstrapping it is much slower;
- evaluate heldout behavior with reference-translation chrF and character F1;
- write aggregate summaries under `artifacts/translation_rigorous_summary/`;
- write a readable final report at `artifacts/translation_rigorous_summary/<run_id>/final_report.md`;
- write layer-wise plots under `artifacts/translation_rigorous_summary/<run_id>/plots/`.

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

DATA_SOURCE=synthetic CUDA_VISIBLE_DEVICES=0 \
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

To re-render the final report for the latest completed run:

```bash
python experiments/report_translation_results.py
python experiments/report_translation_plots.py
```
