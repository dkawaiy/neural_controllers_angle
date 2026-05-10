"""Compare base and LoRA translation-angle probe runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and LoRA angle metrics.")
    parser.add_argument("--base-metrics", required=True)
    parser.add_argument("--lora-metrics", required=True)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "translation_angle_comparison"))
    parser.add_argument("--late-layers", nargs="*", type=int, default=None)
    return parser.parse_args()


def load_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def index_rows(metrics: dict) -> dict[tuple[str, int], dict]:
    return {(row["extractor"], int(row["layer"])): row for row in metrics["rows"]}


def build_comparison_rows(base: dict, lora: dict) -> list[dict]:
    base_rows = index_rows(base)
    lora_rows = index_rows(lora)
    rows = []
    for key in sorted(base_rows):
        if key not in lora_rows:
            continue
        extractor, layer = key
        b = base_rows[key]
        t = lora_rows[key]
        row = {
            "extractor": extractor,
            "layer": layer,
            "base_cosine": b["cosine"],
            "lora_cosine": t["cosine"],
            "delta_cosine": t["cosine"] - b["cosine"],
            "base_angle_degrees": b["angle_degrees"],
            "lora_angle_degrees": t["angle_degrees"],
            "delta_angle_degrees": t["angle_degrees"] - b["angle_degrees"],
            "base_language_auc": b["language_auc"],
            "lora_language_auc": t["language_auc"],
            "base_translation_auc": b["translation_auc"],
            "lora_translation_auc": t["translation_auc"],
        }
        for source_name, source in [("base", b), ("lora", t)]:
            for metric in ["cosine", "angle_degrees"]:
                for suffix in ["ci_low", "ci_high", "boot_mean"]:
                    key = f"{metric}_{suffix}"
                    if key in source:
                        row[f"{source_name}_{key}"] = source[key]
        rows.append(row)
    return rows


def plot_metric(rows: list[dict], output_dir: Path, extractor: str, metric: str, ylabel: str, filename: str) -> None:
    subset = sorted([row for row in rows if row["extractor"] == extractor], key=lambda row: row["layer"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot([row["layer"] for row in subset], [row[f"base_{metric}"] for row in subset], marker="o", label="base")
    ax.plot([row["layer"] for row in subset], [row[f"lora_{metric}"] for row in subset], marker="o", label="translation LoRA")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def plot_delta(rows: list[dict], output_dir: Path, extractor: str, metric: str, ylabel: str, filename: str) -> None:
    subset = sorted([row for row in rows if row["extractor"] == extractor], key=lambda row: row["layer"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.plot([row["layer"] for row in subset], [row[f"delta_{metric}"] for row in subset], marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def summarize_late_layers(rows: list[dict], late_layers: list[int] | None) -> dict:
    if late_layers is None:
        max_layer = max(row["layer"] for row in rows)
        late_layers = list(range(max(0, max_layer - 6), max_layer + 1))
    summary = {"late_layers": late_layers, "by_extractor": {}}
    for extractor in sorted({row["extractor"] for row in rows}):
        subset = [row for row in rows if row["extractor"] == extractor and row["layer"] in late_layers]
        if not subset:
            continue
        summary["by_extractor"][extractor] = {
            "n_layers": len(subset),
            "positive_delta_cosine_layers": sum(row["delta_cosine"] > 0 for row in subset),
            "negative_delta_angle_layers": sum(row["delta_angle_degrees"] < 0 for row in subset),
            "mean_delta_cosine": sum(row["delta_cosine"] for row in subset) / len(subset),
            "mean_delta_angle_degrees": sum(row["delta_angle_degrees"] for row in subset) / len(subset),
            "layer_values": subset,
        }
    return summary


def main() -> None:
    args = parse_args()
    base = load_metrics(args.base_metrics)
    lora = load_metrics(args.lora_metrics)
    rows = build_comparison_rows(base, lora)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for extractor in sorted({row["extractor"] for row in rows}):
        plot_metric(rows, output_dir, extractor, "angle_degrees", "Angle (degrees)", f"{extractor}_angle_comparison.png")
        plot_metric(rows, output_dir, extractor, "cosine", "Cosine similarity", f"{extractor}_cosine_comparison.png")
        plot_delta(rows, output_dir, extractor, "angle_degrees", "Delta angle, LoRA - base", f"{extractor}_delta_angle.png")
        plot_delta(rows, output_dir, extractor, "cosine", "Delta cosine, LoRA - base", f"{extractor}_delta_cosine.png")

    late_summary = summarize_late_layers(rows, args.late_layers)
    with (output_dir / "late_layer_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(late_summary, handle, indent=2, ensure_ascii=False)

    summary = {
        "run_id": run_id,
        "base_metrics": args.base_metrics,
        "lora_metrics": args.lora_metrics,
        "base_model_state": base.get("model_state"),
        "lora_model_state": lora.get("model_state"),
        "late_layer_summary": late_summary,
        "rows": rows,
        "output_dir": str(output_dir),
    }
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
