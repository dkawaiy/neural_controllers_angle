# First-Run Plan

## Current Repo Shape

- The upstream repo is a paper-support library for neural controllers, with notebook-first examples.
- Core code already supports direction methods such as RFM, logistic probing, mean difference, PCA, and linear probes.
- Several helper paths assume CUDA through `.cuda()` calls, so CPU-first local experiments should use a small wrapper path first.
- The project did not include a checked-in dependency file, so this setup adds local and GPU requirement files.

## Local Smoke Test

Goal: verify the machine can load a decoder-only model, extract hidden states, train simple concept directions, and save artifacts.

Default model:

```text
EleutherAI/pythia-14m
```

This is about 0.014B parameters and is small enough for CPU smoke testing. It is only for pipeline validation, not for the final translation/factual experiments.

Command:

```bash
.venv/bin/python experiments/tiny_smoke_test.py
```

Expected outputs:

```text
artifacts/smoke/{run_id}/metrics.json
artifacts/smoke/{run_id}/activations.npz
artifacts/smoke/{run_id}/metadata.jsonl
```

Verified locally:

- `experiments/tiny_smoke_test.py` runs with `EleutherAI/pythia-14m` on CPU.
- The upstream `NeuralController` entry point runs with `control_method="mean_difference"` on CPU for the same tiny model.
- A small compatibility patch now lets activation extraction infer layer counts from common decoder-only model families, including GPT-NeoX/Pythia.

## First Translation Angle Probe

Model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Dataset:

```text
data/translation_mini/en_zh_mini.jsonl
```

Command:

```bash
.venv/bin/python experiments/translation_angle_probe.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --batch-size 4 \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
```

Current run:

```text
artifacts/translation_angle/20260510_215105
```

Quick observation:

- Early and middle layers are close to orthogonal for `v_lang` vs `v_trans`.
- Late layers show larger cosine similarity.
- With mean difference, layer 23 reaches cosine `0.683`, angle `46.95` degrees.
- With logistic, layer 23 reaches cosine `0.379`, angle `67.71` degrees.
- This is only a base-model probe on a tiny in-sample dataset; it is a useful sanity signal, not evidence for the fine-tuning hypothesis yet.

## First Base vs Translation-LoRA Comparison

Tiny LoRA:

```text
artifacts/loras/translation_en_zh/20260510_215743
```

Training setup:

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Data: 36 EN-ZH pairs
- Device: CPU
- LoRA target modules: `q_proj`, `v_proj`
- LoRA rank: 4
- Optimizer steps: 20
- Trainable parameters: 270,336

Comparison run:

```text
artifacts/translation_angle_comparison/20260510_215836
```

Quick observation:

- Late layers 17-23 all move toward stronger alignment for both mean-diff and logistic.
- Mean-diff late-layer average: delta cosine `+0.048`, delta angle `-3.13` degrees.
- Logistic late-layer average: delta cosine `+0.0625`, delta angle `-3.86` degrees.
- Layer 23 mean-diff: angle goes from `46.95` to `41.34` degrees.
- Layer 23 logistic: angle goes from `67.71` to `60.11` degrees.
- Early/middle layers are mixed, which is actually useful: the effect is layer-specific, concentrated late.
- This is still a tiny in-sample result. Next version needs train/test split, random-label LoRA, seed repeats, and bootstrap.

## Rigorous Translation Geometry Upgrade

Added a stricter pipeline around the preliminary signal:

```text
experiments/generate_synthetic_translation_data.py
experiments/train_translation_lora.py
experiments/translation_angle_probe.py
experiments/evaluate_translation_behavior.py
experiments/compare_translation_angles.py
experiments/summarize_translation_controls.py
experiments/run_rigorous_translation_geometry.sh
```

The synthetic data has four splits:

```text
translation_finetune_train
translation_geometry_train
translation_geometry_test
translation_behavior_test
```

The stricter angle probe can now:

- fit/report separate geometry splits with `--fit-split` and `--test-split`;
- bootstrap test-split angles with `--bootstrap-samples`;
- run `mean_diff`, `logistic`, and native `rfm` extractors by default;
- save both `fit_rows` and independent test `rows`;
- preserve CI columns in comparison CSVs.

`rfm` calls the upstream xRFM `RFM` implementation directly and takes the top eigenvector of `agop_best_model` as the concept direction. `mlp_agop` remains available only as a Mac/MPS-safe fallback approximation.

The LoRA trainer can now:

- train only selected splits via `--train-splits`;
- run a matched random-target control via `--random-targets`.

The behavior evaluator measures synthetic target-term recall on heldout behavior examples. In a smoke run, base Qwen2.5-0.5B had `0.0` term recall on synthetic mappings, which is useful because it creates measurable room for LoRA behavior improvement.

Full MPS rigorous run:

```bash
bash experiments/run_rigorous_translation_geometry.sh
```

Quick smoke:

```bash
SEEDS="42" MAX_STEPS=5 BOOTSTRAP=20 bash experiments/run_rigorous_translation_geometry.sh
```

## Phase 1: Infrastructure

1. Keep environment setup reproducible with `requirements-local.txt` and `requirements-gpu-cu118.txt`.
2. Route Hugging Face and Matplotlib caches to project-local ignored folders.
3. Add a model/tokenizer loader that accepts base models and optional LoRA adapters.
4. Add config files for local smoke, Qwen 7B translation, and factual triples.
5. Keep paths, seeds, model names, layers, extractors, and batch sizes in config.

## Phase 2: Translation Geometry MVP

1. Generate balanced English/Chinese language-classification samples.
2. Generate translation-active vs control prompts with matched templates.
3. Fine-tune LoRA on English to Chinese translation examples.
4. Extract last-token activations for base and LoRA models.
5. Fit mean-diff, logistic, and native RFM vectors per selected layer.
6. Compare `cos(v_lang, v_trans)` before and after fine-tuning.
7. Run random-label controls and bootstrap confidence intervals.

## Phase 3: Factual TransE Experiment

1. Generate synthetic triples with train, seen-test, heldout, corrupted-tail, and relation-shuffled splits.
2. Fine-tune LoRA on factual QA/completion examples.
3. Verify behavioral learning before geometry analysis.
4. Extract head, relation, and tail concept vectors.
5. Compute normalized TransE residual and cosine alternatives.
6. Compare true triples against controls across layers.

## Migration Notes

- Local runs should use tiny models and CPU-safe code paths.
- GPU runs should use `requirements-gpu-cu118.txt`, `device_map=auto`, and QLoRA where appropriate.
- Avoid hard-coding absolute paths in experiment configs.
- Save every run under a unique `run_id` and include config plus git commit hash in reports.
