"""Evaluate synthetic lexicon translation behavior by target-term recall."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".hf_cache" / "datasets"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate behavior on synthetic translation examples.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--model-state", default=None)
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data" / "translation_synthetic" / "synthetic_lexicon_en_zh.jsonl"))
    parser.add_argument("--split", default="translation_behavior_test")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "translation_behavior"))
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


def read_pairs(path: str, split: str) -> list[dict]:
    pairs = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            pair = json.loads(line)
            if pair.get("split") == split:
                pairs.append(pair)
    if not pairs:
        raise ValueError(f"No examples found for split {split!r}")
    return pairs


def format_prompt(tokenizer, english: str) -> str:
    messages = [
        {"role": "system", "content": "You are a precise English to Chinese translation assistant."},
        {"role": "user", "content": f"Translate this English sentence into Chinese:\n{english}"},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Translate this English sentence into Chinese:\n{english}\nChinese:"


def generate(model, tokenizer, prompt: str, device: torch.device, max_length: int, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, encoded["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def zh_char_counter(text: str) -> Counter:
    return Counter(ch for ch in text if "\u4e00" <= ch <= "\u9fff")


def char_f1(reference: str, output: str) -> float:
    ref_counts = zh_char_counter(reference)
    out_counts = zh_char_counter(output)
    if not ref_counts or not out_counts:
        return 0.0
    overlap = sum((ref_counts & out_counts).values())
    precision = overlap / max(1, sum(out_counts.values()))
    recall = overlap / max(1, sum(ref_counts.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def zh_chars(text: str) -> list[str]:
    return [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]


def char_ngrams(chars: list[str], n: int) -> Counter:
    if len(chars) < n:
        return Counter()
    return Counter("".join(chars[idx : idx + n]) for idx in range(len(chars) - n + 1))


def chrf(reference: str, output: str, max_order: int = 6, beta: float = 2.0) -> float:
    ref_chars = zh_chars(reference)
    out_chars = zh_chars(output)
    precisions, recalls = [], []
    for order in range(1, max_order + 1):
        ref_counts = char_ngrams(ref_chars, order)
        out_counts = char_ngrams(out_chars, order)
        if not ref_counts or not out_counts:
            continue
        overlap = sum((ref_counts & out_counts).values())
        precisions.append(overlap / max(1, sum(out_counts.values())))
        recalls.append(overlap / max(1, sum(ref_counts.values())))
    if not precisions or not recalls:
        return 0.0
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    beta_sq = beta * beta
    if precision + recall == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def score_output(pair: dict, output: str) -> dict:
    expected_terms = [term["zh"] for term in pair.get("terms", [])]
    hits = [term for term in expected_terms if term in output]
    reference = pair["zh"]
    reference_char_f1 = char_f1(reference, output)
    return {
        "expected_terms": expected_terms,
        "hit_terms": hits,
        "term_recall": len(hits) / max(1, len(expected_terms)),
        "all_terms_hit": len(hits) == len(expected_terms),
        "has_terms": bool(expected_terms),
        "reference_char_f1": reference_char_f1,
        "reference_chrf": chrf(reference, output),
        "reference_contained": reference in output,
    }


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    pairs = read_pairs(args.data_path, args.split)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, trust_remote_code=True).to(device)
    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path).to(device)
    model.eval()

    rows = []
    for pair in pairs:
        prompt = format_prompt(tokenizer, pair["en"])
        output = generate(model, tokenizer, prompt, device, args.max_length, args.max_new_tokens)
        score = score_output(pair, output)
        rows.append(
            {
                "id": pair["id"],
                "split": pair.get("split"),
                "en": pair["en"],
                "reference_zh": pair["zh"],
                "output": output,
                "expected_terms": "|".join(score["expected_terms"]),
                "hit_terms": "|".join(score["hit_terms"]),
                "term_recall": score["term_recall"],
                "all_terms_hit": score["all_terms_hit"],
                "has_terms": score["has_terms"],
                "reference_char_f1": score["reference_char_f1"],
                "reference_chrf": score["reference_chrf"],
                "reference_contained": score["reference_contained"],
            }
        )

    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "behavior.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "run_id": run_id,
        "model_id": args.model_id,
        "model_state": args.model_state or ("translation_lora" if args.lora_path else "base"),
        "lora_path": args.lora_path,
        "data_path": args.data_path,
        "split": args.split,
        "device": str(device),
        "dtype": str(dtype),
        "num_examples": len(rows),
        "mean_term_recall": sum(row["term_recall"] for row in rows) / len(rows),
        "all_terms_hit_rate": sum(bool(row["all_terms_hit"]) for row in rows) / len(rows),
        "mean_reference_char_f1": sum(row["reference_char_f1"] for row in rows) / len(rows),
        "mean_reference_chrf": sum(row["reference_chrf"] for row in rows) / len(rows),
        "reference_contained_rate": sum(bool(row["reference_contained"]) for row in rows) / len(rows),
        "has_term_annotations": any(bool(row["has_terms"]) for row in rows),
        "rows": rows,
        "output_dir": str(output_dir),
    }
    with (output_dir / "behavior_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
