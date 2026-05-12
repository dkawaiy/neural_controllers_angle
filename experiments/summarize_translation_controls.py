"""Summarize real-vs-control translation geometry runs across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate translation geometry control comparisons.")
    parser.add_argument("--real-comparisons", nargs="*", default=[])
    parser.add_argument("--random-comparisons", nargs="*", default=[])
    parser.add_argument("--real-behavior", nargs="*", default=[])
    parser.add_argument("--random-behavior", nargs="*", default=[])
    parser.add_argument("--late-layers", nargs="*", type=int, default=None)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "translation_rigorous_summary"))
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_comparison(path: str, control_type: str, seed_idx: int, late_layers: list[int] | None) -> list[dict]:
    data = load_json(path)
    rows = data["rows"]
    if late_layers is None:
        max_layer = max(row["layer"] for row in rows)
        late_layers = list(range(max(0, max_layer - 6), max_layer + 1))
    summaries = []
    for extractor in sorted({row["extractor"] for row in rows}):
        subset = [row for row in rows if row["extractor"] == extractor and row["layer"] in late_layers]
        if not subset:
            continue
        summaries.append(
            {
                "control_type": control_type,
                "seed_index": seed_idx,
                "extractor": extractor,
                "comparison_path": path,
                "late_layers": " ".join(str(layer) for layer in late_layers),
                "n_layers": len(subset),
                "positive_delta_cosine_layers": sum(row["delta_cosine"] > 0 for row in subset),
                "negative_delta_angle_layers": sum(row["delta_angle_degrees"] < 0 for row in subset),
                "mean_delta_cosine": sum(row["delta_cosine"] for row in subset) / len(subset),
                "mean_delta_angle_degrees": sum(row["delta_angle_degrees"] for row in subset) / len(subset),
            }
        )
    return summaries


def summarize_behavior(paths: list[str], control_type: str) -> list[dict]:
    rows = []
    for seed_idx, path in enumerate(paths):
        data = load_json(path)
        rows.append(
            {
                "control_type": control_type,
                "seed_index": seed_idx,
                "behavior_path": path,
                "mean_term_recall": data["mean_term_recall"],
                "all_terms_hit_rate": data["all_terms_hit_rate"],
                "mean_reference_char_f1": data.get("mean_reference_char_f1"),
                "reference_contained_rate": data.get("reference_contained_rate"),
                "has_term_annotations": data.get("has_term_annotations"),
                "num_examples": data["num_examples"],
            }
        )
    return rows


def aggregate_geometry(rows: list[dict]) -> dict:
    aggregate = {}
    for control_type in sorted({row["control_type"] for row in rows}):
        aggregate[control_type] = {}
        for extractor in sorted({row["extractor"] for row in rows if row["control_type"] == control_type}):
            subset = [row for row in rows if row["control_type"] == control_type and row["extractor"] == extractor]
            delta_cos = [row["mean_delta_cosine"] for row in subset]
            delta_angle = [row["mean_delta_angle_degrees"] for row in subset]
            aggregate[control_type][extractor] = {
                "n_seeds": len(subset),
                "mean_delta_cosine": statistics.mean(delta_cos),
                "mean_delta_angle_degrees": statistics.mean(delta_angle),
                "stdev_delta_cosine": statistics.stdev(delta_cos) if len(delta_cos) > 1 else 0.0,
                "stdev_delta_angle_degrees": statistics.stdev(delta_angle) if len(delta_angle) > 1 else 0.0,
                "all_seed_values": subset,
            }
    return aggregate


def aggregate_behavior(rows: list[dict]) -> dict:
    aggregate = {}
    for control_type in sorted({row["control_type"] for row in rows}):
        subset = [row for row in rows if row["control_type"] == control_type]
        recalls = [row["mean_term_recall"] for row in subset]
        hit_rates = [row["all_terms_hit_rate"] for row in subset]
        char_f1s = [row["mean_reference_char_f1"] for row in subset if row.get("mean_reference_char_f1") is not None]
        contained_rates = [row["reference_contained_rate"] for row in subset if row.get("reference_contained_rate") is not None]
        aggregate[control_type] = {
            "n_seeds": len(subset),
            "mean_term_recall": statistics.mean(recalls),
            "mean_all_terms_hit_rate": statistics.mean(hit_rates),
            "mean_reference_char_f1": statistics.mean(char_f1s) if char_f1s else None,
            "mean_reference_contained_rate": statistics.mean(contained_rates) if contained_rates else None,
            "all_seed_values": subset,
        }
    return aggregate


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    geometry_rows = []
    for idx, path in enumerate(args.real_comparisons):
        geometry_rows.extend(summarize_comparison(path, "real", idx, args.late_layers))
    for idx, path in enumerate(args.random_comparisons):
        geometry_rows.extend(summarize_comparison(path, "random_target", idx, args.late_layers))

    behavior_rows = []
    behavior_rows.extend(summarize_behavior(args.real_behavior, "real"))
    behavior_rows.extend(summarize_behavior(args.random_behavior, "random_target"))

    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "geometry_summary.csv", geometry_rows)
    write_csv(output_dir / "behavior_summary.csv", behavior_rows)

    summary = {
        "run_id": run_id,
        "geometry": aggregate_geometry(geometry_rows) if geometry_rows else {},
        "behavior": aggregate_behavior(behavior_rows) if behavior_rows else {},
        "geometry_rows": geometry_rows,
        "behavior_rows": behavior_rows,
        "output_dir": str(output_dir),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
