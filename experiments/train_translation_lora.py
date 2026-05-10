"""Tiny CPU-friendly LoRA fine-tune for EN-ZH translation."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".hf_cache" / "datasets"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class TranslationDataset(Dataset):
    def __init__(self, examples: list[dict[str, torch.Tensor]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a tiny LoRA adapter for EN-ZH translation.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data" / "translation_mini" / "en_zh_mini.jsonl"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "loras" / "translation_en_zh"))
    parser.add_argument("--train-splits", nargs="*", default=None, help="Optional split names to use for LoRA training.")
    parser.add_argument("--random-targets", action="store_true", help="Shuffle target translations as a random-label control.")
    parser.add_argument("--control-type", default=None, help="Optional label written to train_summary.json.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-modules", nargs="*", default=["q_proj", "v_proj"])
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_dtype(dtype_arg: str, device: torch.device):
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def read_pairs(path: str) -> list[dict[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def filter_pairs(pairs: list[dict[str, str]], splits: list[str] | None) -> list[dict[str, str]]:
    if not splits:
        return pairs
    split_set = set(splits)
    filtered = [pair for pair in pairs if pair.get("split") in split_set]
    if not filtered:
        raise ValueError(f"No examples found for splits: {sorted(split_set)}")
    return filtered


def maybe_shuffle_targets(pairs: list[dict[str, str]], seed: int, random_targets: bool) -> list[dict[str, str]]:
    if not random_targets:
        return pairs
    rng = random.Random(seed)
    shuffled_targets = [pair["zh"] for pair in pairs]
    rng.shuffle(shuffled_targets)
    if len(shuffled_targets) > 1 and all(target == pair["zh"] for pair, target in zip(pairs, shuffled_targets)):
        shuffled_targets = shuffled_targets[1:] + shuffled_targets[:1]
    shuffled = []
    for pair, target in zip(pairs, shuffled_targets):
        new_pair = dict(pair)
        new_pair["true_zh"] = pair["zh"]
        new_pair["zh"] = target
        new_pair["target_control"] = "random_target"
        shuffled.append(new_pair)
    return shuffled


def format_prompt(tokenizer, english: str) -> str:
    messages = [
        {"role": "system", "content": "You are a precise English to Chinese translation assistant."},
        {"role": "user", "content": f"Translate this English sentence into Chinese:\n{english}"},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Translate this English sentence into Chinese:\n{english}\nChinese:"


def make_examples(tokenizer, pairs: list[dict[str, str]], max_length: int) -> list[dict[str, torch.Tensor]]:
    examples = []
    eos = tokenizer.eos_token or ""
    for pair in pairs:
        prompt = format_prompt(tokenizer, pair["en"])
        answer = pair["zh"] + eos
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = tokenizer(prompt + answer, add_special_tokens=False).input_ids[:max_length]
        labels = full_ids.copy()
        mask_len = min(len(prompt_ids), len(labels))
        labels[:mask_len] = [-100] * mask_len
        if all(label == -100 for label in labels):
            continue
        examples.append(
            {
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        )
    return examples


def collate_batch(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        pad_len = max_len - item["input_ids"].shape[0]
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)

    all_pairs = read_pairs(args.data_path)
    pairs = filter_pairs(all_pairs, args.train_splits)
    pairs = maybe_shuffle_targets(pairs, args.seed, args.random_targets)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, trust_remote_code=True)
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config).to(device)
    model.train()
    model.print_trainable_parameters()

    examples = make_examples(tokenizer, pairs, args.max_length)
    if not examples:
        raise ValueError("No trainable examples were created.")
    dataset = TranslationDataset(examples)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    losses = []
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    start_time = time.time()
    while step < args.max_steps:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            micro_step += 1
            if micro_step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                losses.append(float(loss.detach().cpu()) * args.grad_accum)
                print(json.dumps({"step": step, "loss": losses[-1]}, ensure_ascii=False))
                if step >= args.max_steps:
                    break

    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
        "run_id": run_id,
        "model_id": args.model_id,
        "data_path": args.data_path,
        "train_splits": args.train_splits,
        "control_type": args.control_type or ("random_target" if args.random_targets else "real"),
        "random_targets": args.random_targets,
        "device": str(device),
        "dtype": str(dtype),
        "num_pairs": len(pairs),
        "num_examples": len(examples),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "losses": losses,
        "elapsed_seconds": time.time() - start_time,
        "output_dir": str(output_dir),
    }
    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
