"""Prepare real EN-ZH translation pairs for geometry experiments."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".hf_cache" / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(REPO_ROOT / ".hf_cache" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(REPO_ROOT / ".hf_cache" / "transformers"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import DownloadConfig, load_dataset


SPLIT_ORDER = [
    "translation_finetune_train",
    "translation_geometry_train",
    "translation_geometry_test",
    "translation_behavior_test",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/sample a real EN-ZH translation jsonl.")
    parser.add_argument("--out-path", default=str(REPO_ROOT / "data" / "translation_real" / "en_zh_swaption20k_sample.jsonl"))
    parser.add_argument("--dataset", default="swaption20k", choices=["swaption20k", "custom_hf"])
    parser.add_argument("--dataset-name", default=None, help="HF dataset name for --dataset custom_hf.")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--source-key", default="en", help="Source key inside a translation dict for custom_hf.")
    parser.add_argument("--target-key", default="zh", help="Target key inside a translation dict for custom_hf.")
    parser.add_argument("--n-finetune", type=int, default=1000)
    parser.add_argument("--n-geometry-train", type=int, default=500)
    parser.add_argument("--n-geometry-test", type=int, default=500)
    parser.add_argument("--n-behavior-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-en-chars", type=int, default=12)
    parser.add_argument("--max-en-chars", type=int, default=260)
    parser.add_argument("--min-zh-chars", type=int, default=4)
    parser.add_argument("--max-zh-chars", type=int, default=220)
    parser.add_argument("--local-files-only", action="store_true", help="Use cached HF files only.")
    return parser.parse_args()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def remove_prefix(text: str, prefix: str) -> str:
    text = text.strip()
    if text.lower().startswith(prefix):
        return text[len(prefix) :].strip()
    return text


def valid_pair(en: str, zh: str, args: argparse.Namespace) -> bool:
    if not (args.min_en_chars <= len(en) <= args.max_en_chars):
        return False
    if not (args.min_zh_chars <= len(zh) <= args.max_zh_chars):
        return False
    if en == zh:
        return False
    return bool(re.search(r"[A-Za-z]", en)) and bool(re.search(r"[\u4e00-\u9fff]", zh))


def load_swaption20k(args: argparse.Namespace) -> list[dict[str, str]]:
    dataset = load_dataset(
        "swaption2009/20k-en-zh-translation-pinyin-hsk",
        split=args.hf_split,
        data_dir="",
        download_config=DownloadConfig(local_files_only=args.local_files_only),
        download_mode="reuse_dataset_if_exists",
        verification_mode="no_checks",
    )
    rows = [clean_text(row["text"]) for row in dataset if clean_text(row.get("text", ""))]
    pairs = []
    seen = set()

    for start in range(0, len(rows), 5):
        group = rows[start : start + 5]
        english = next((remove_prefix(item, "english:") for item in group if item.lower().startswith("english:")), None)
        chinese = next((remove_prefix(item, "mandarin:") for item in group if item.lower().startswith("mandarin:")), None)
        if not english or not chinese:
            continue
        english = clean_text(english)
        chinese = clean_text(chinese)
        key = (english, chinese)
        if key in seen or not valid_pair(english, chinese, args):
            continue
        seen.add(key)
        pairs.append({"en": english, "zh": chinese})

    return pairs


def load_custom_hf(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.dataset_name:
        raise ValueError("--dataset-name is required for --dataset custom_hf")
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.hf_split,
        download_config=DownloadConfig(local_files_only=args.local_files_only),
        download_mode="reuse_dataset_if_exists",
        verification_mode="no_checks",
    )
    pairs = []
    seen = set()
    for row in dataset:
        if "translation" in row and isinstance(row["translation"], dict):
            trans = row["translation"]
            en = trans.get(args.source_key)
            zh = trans.get(args.target_key)
        else:
            en = row.get(args.source_key)
            zh = row.get(args.target_key)
        if not en or not zh:
            continue
        en = clean_text(en)
        zh = clean_text(zh)
        key = (en, zh)
        if key in seen or not valid_pair(en, zh, args):
            continue
        seen.add(key)
        pairs.append({"en": en, "zh": zh})
    return pairs


def assign_splits(pairs: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, object]]:
    split_sizes = {
        "translation_finetune_train": args.n_finetune,
        "translation_geometry_train": args.n_geometry_train,
        "translation_geometry_test": args.n_geometry_test,
        "translation_behavior_test": args.n_behavior_test,
    }
    required = sum(split_sizes.values())
    if len(pairs) < required:
        raise ValueError(f"Need {required} valid pairs but found only {len(pairs)}.")

    rng = random.Random(args.seed)
    sampled = pairs[:]
    rng.shuffle(sampled)
    examples = []
    cursor = 0
    for split in SPLIT_ORDER:
        for pair in sampled[cursor : cursor + split_sizes[split]]:
            examples.append(
                {
                    "id": f"real_{len(examples):06d}",
                    "split": split,
                    "en": pair["en"],
                    "zh": pair["zh"],
                    "terms": [],
                    "source_dataset": args.dataset_name or args.dataset,
                }
            )
        cursor += split_sizes[split]
    return examples


def main() -> None:
    args = parse_args()
    if args.local_files_only:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    if args.dataset == "swaption20k":
        pairs = load_swaption20k(args)
    else:
        pairs = load_custom_hf(args)
    examples = assign_splits(pairs, args)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    split_sizes = {split: sum(example["split"] == split for example in examples) for split in SPLIT_ORDER}
    summary = {
        "out_path": str(out_path),
        "dataset": args.dataset_name or args.dataset,
        "hf_split": args.hf_split,
        "seed": args.seed,
        "num_valid_pairs_seen": len(pairs),
        "num_examples": len(examples),
        "split_sizes": split_sizes,
        "preview": examples[:3],
    }
    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
