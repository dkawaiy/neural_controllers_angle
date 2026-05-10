"""Generate split synthetic EN-ZH translation data with artificial lexicon mappings."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


LEXICON = [
    ("glorp", "星棱体"),
    ("nareth", "暗港"),
    ("virel process", "微光过程"),
    ("soma key", "素玛钥"),
    ("kelin archive", "柯林档案"),
    ("orvex stone", "奥维石"),
    ("tarn field", "塔恩场"),
    ("luma gate", "流明门"),
    ("prax engine", "普拉克斯引擎"),
    ("zelic map", "泽利克图"),
    ("mavon thread", "马文线"),
    ("irid lens", "虹晶镜"),
]


TEMPLATES = [
    ("The {en0} is stored beside the {en1}.", "{zh0}被存放在{zh1}旁边。"),
    ("Engineers tested the {en0} before opening the {en1}.", "工程师在打开{zh1}之前测试了{zh0}。"),
    ("The report says the {en0} controls the {en1}.", "报告说{zh0}控制着{zh1}。"),
    ("A small team moved the {en0} through the {en1}.", "一个小团队把{zh0}移过了{zh1}。"),
    ("The old manual describes how the {en0} reacts to the {en1}.", "旧手册描述了{zh0}如何对{zh1}产生反应。"),
    ("During the drill, the {en0} was connected to the {en1}.", "演练期间，{zh0}被连接到{zh1}。"),
    ("The scientist compared the {en0} with the {en1}.", "科学家比较了{zh0}和{zh1}。"),
    ("They placed the {en0} inside the chamber near the {en1}.", "他们把{zh0}放进靠近{zh1}的舱室里。"),
    ("The {en0} failed when the {en1} became unstable.", "当{zh1}变得不稳定时，{zh0}失效了。"),
    ("A warning appeared after the {en0} touched the {en1}.", "在{zh0}接触{zh1}后出现了警告。"),
]


SPLIT_SIZES = {
    "translation_finetune_train": 96,
    "translation_geometry_train": 48,
    "translation_geometry_test": 48,
    "translation_behavior_test": 24,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic translation data with fixed splits.")
    parser.add_argument("--out-path", default=str(REPO_ROOT / "data" / "translation_synthetic" / "synthetic_lexicon_en_zh.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-finetune", type=int, default=SPLIT_SIZES["translation_finetune_train"])
    parser.add_argument("--n-geometry-train", type=int, default=SPLIT_SIZES["translation_geometry_train"])
    parser.add_argument("--n-geometry-test", type=int, default=SPLIT_SIZES["translation_geometry_test"])
    parser.add_argument("--n-behavior-test", type=int, default=SPLIT_SIZES["translation_behavior_test"])
    return parser.parse_args()


def make_example(idx: int, split: str, rng: random.Random) -> dict[str, object]:
    (en0, zh0), (en1, zh1) = rng.sample(LEXICON, 2)
    en_template, zh_template = rng.choice(TEMPLATES)
    en = en_template.format(en0=en0, en1=en1)
    zh = zh_template.format(zh0=zh0, zh1=zh1)
    return {
        "id": f"synthetic_{idx:05d}",
        "split": split,
        "en": en,
        "zh": zh,
        "terms": [
            {"en": en0, "zh": zh0},
            {"en": en1, "zh": zh1},
        ],
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    split_sizes = {
        "translation_finetune_train": args.n_finetune,
        "translation_geometry_train": args.n_geometry_train,
        "translation_geometry_test": args.n_geometry_test,
        "translation_behavior_test": args.n_behavior_test,
    }
    examples = []
    idx = 0
    for split, count in split_sizes.items():
        for _ in range(count):
            examples.append(make_example(idx, split, rng))
            idx += 1

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "out_path": str(out_path),
                "seed": args.seed,
                "split_sizes": split_sizes,
                "num_examples": len(examples),
                "lexicon": [{"en": en, "zh": zh} for en, zh in LEXICON],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps({"out_path": str(out_path), "summary_path": str(summary_path), "num_examples": len(examples)}, indent=2))


if __name__ == "__main__":
    main()
