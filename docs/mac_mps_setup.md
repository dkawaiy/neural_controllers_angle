# Mac MPS Setup

## What Happened

The machine has an Apple M3 GPU:

```text
Apple M3, 10 GPU cores, Metal supported
```

The original local environment used `torch==2.4.0` to match the upstream README. On this macOS 26 system that wheel reports:

```text
mps_built=True
mps_available=False
```

It also raises an OS-version error when creating an MPS tensor. A separate `.venv-mps` environment has therefore been created with a newer PyTorch wheel.

Inside the Codex tool sandbox, native Metal currently returns:

```text
metal_device nil
```

So the hardware exists, but this particular sandboxed process cannot create a Metal device. Run the checks below from a normal macOS Terminal to confirm MPS access outside Codex.

## Environment

```bash
cd /Users/duanyong/projects/findings/neural_controllers
source .venv-mps/bin/activate

export HF_HOME=.hf_cache
export HF_DATASETS_CACHE=.hf_cache/datasets
export MPLCONFIGDIR=.mpl_cache
export XDG_CACHE_HOME=.cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

The MPS requirements are stored in:

```text
requirements-mps.txt
```

To rebuild:

```bash
python3 -m venv .venv-mps
.venv-mps/bin/python -m pip install --upgrade pip setuptools wheel
.venv-mps/bin/python -m pip install -r requirements-mps.txt
```

## Verify GPU

Run:

```bash
.venv-mps/bin/python experiments/check_mac_gpu.py
```

Healthy output should include:

```text
metal_device Apple M3
mps_available: True
mps_tensor_device: mps:0
```

If `system_profiler` shows Apple M3 but `metal_device nil`, the current process is blocked from Metal. Use a normal Terminal app, not a sandboxed runner.

## Run MPS Experiments

One-shot comparison from the normal macOS Terminal:

```bash
bash experiments/run_mps_translation_comparison.sh
```

Rigorous control run with synthetic lexicon data, real LoRA, random-target LoRA, seeds, bootstrap, and behavior evaluation:

```bash
bash experiments/run_rigorous_translation_geometry.sh
```

Quick smoke version:

```bash
SEEDS="42" MAX_STEPS=5 BOOTSTRAP=20 bash experiments/run_rigorous_translation_geometry.sh
```

Manual commands are below.

Angle probe:

```bash
.venv-mps/bin/python experiments/translation_angle_probe.py \
  --device mps \
  --dtype float16 \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --model-state base_mps \
  --batch-size 4 \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
```

Tiny LoRA:

```bash
.venv-mps/bin/python experiments/train_translation_lora.py \
  --device mps \
  --dtype float16 \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --batch-size 1 \
  --grad-accum 4 \
  --max-steps 20 \
  --learning-rate 5e-4 \
  --lora-r 4 \
  --lora-alpha 8 \
  --target-modules q_proj v_proj
```

Then rerun the angle probe with:

```bash
--lora-path artifacts/loras/translation_en_zh/{run_id}
```
