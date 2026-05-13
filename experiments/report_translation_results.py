"""Create a readable final report for rigorous translation geometry runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "artifacts" / "translation_rigorous_summary"


T_CRITICAL_95 = {
    1: None,
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final translation geometry results.")
    parser.add_argument("--summary-dir", default=None, help="Run directory under artifacts/translation_rigorous_summary. Defaults to latest.")
    parser.add_argument("--summary-root", default=str(DEFAULT_SUMMARY_ROOT))
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-layer-csv", default=None)
    parser.add_argument(
        "--layer-sets",
        nargs="*",
        default=["late_17_23:17-23", "pre_output_17_22:17-22", "final_23:23"],
        help="Named layer sets, e.g. late_17_23:17-23 final_23:23.",
    )
    return parser.parse_args()


def latest_summary_dir(summary_root: Path) -> Path:
    candidates = [path for path in summary_root.glob("*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No summary runs found under {summary_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def coerce_value(value: str) -> Any:
    if value == "":
        return None
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if any(char in value for char in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: coerce_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None]


def describe(values: list[float]) -> dict[str, Any]:
    values = numeric(values)
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "stdev": None, "sem": None, "ci95_low": None, "ci95_high": None, "values": []}
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if n > 1 else 0.0
    sem = stdev / math.sqrt(n) if n > 1 else 0.0
    tcrit = T_CRITICAL_95.get(n, 1.96)
    if tcrit is None:
        ci_low = ci_high = None
    else:
        ci_low = mean - tcrit * sem
        ci_high = mean + tcrit * sem
    return {
        "n": n,
        "mean": mean,
        "stdev": stdev,
        "sem": sem,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "values": values,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def fmt_ci(stats: dict[str, Any], digits: int = 3) -> str:
    if stats["mean"] is None:
        return "NA"
    if stats["ci95_low"] is None:
        return f"{fmt(stats['mean'], digits)}"
    return f"{fmt(stats['mean'], digits)} [{fmt(stats['ci95_low'], digits)}, {fmt(stats['ci95_high'], digits)}]"


def group_rows(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    return grouped


def parse_layer_sets(items: list[str]) -> dict[str, list[int]]:
    layer_sets = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Layer set must be name:layers, got {item!r}")
        name, spec = item.split(":", 1)
        layers = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                layers.extend(range(int(start), int(end) + 1))
            else:
                layers.append(int(part))
        if not layers:
            raise ValueError(f"Layer set {name!r} has no layers.")
        layer_sets[name] = sorted(set(layers))
    return layer_sets


def load_detailed_layer_rows(geometry_rows: list[dict[str, Any]], summary_dir: Path) -> list[dict[str, Any]]:
    detailed = []
    seen_paths = set()
    for geometry_row in geometry_rows:
        path_value = geometry_row.get("comparison_path")
        if not path_value or path_value in seen_paths:
            continue
        seen_paths.add(path_value)
        path = Path(str(path_value))
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            path = summary_dir / str(path_value)
        if not path.exists():
            continue
        comparison = load_json(path)
        control_type = next(
            (row.get("control_type") for row in geometry_rows if row.get("comparison_path") == path_value),
            None,
        )
        seed_index = next(
            (row.get("seed_index") for row in geometry_rows if row.get("comparison_path") == path_value),
            None,
        )
        for row in comparison.get("rows", []):
            detailed.append(
                {
                    "control_type": control_type,
                    "seed_index": seed_index,
                    "extractor": row["extractor"],
                    "layer": int(row["layer"]),
                    "delta_angle_degrees": float(row["delta_angle_degrees"]),
                    "delta_cosine": float(row["delta_cosine"]),
                    "base_angle_degrees": float(row["base_angle_degrees"]),
                    "lora_angle_degrees": float(row["lora_angle_degrees"]),
                    "base_cosine": float(row["base_cosine"]),
                    "lora_cosine": float(row["lora_cosine"]),
                }
            )
    return detailed


def summarize_behavior(behavior_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric = "mean_reference_chrf"
    if not any(row.get(metric) is not None for row in behavior_rows):
        metric = "mean_reference_char_f1"
    if not any(row.get(metric) is not None for row in behavior_rows):
        metric = "mean_term_recall"
    grouped = group_rows(behavior_rows, "control_type")
    by_control = {}
    for (control_type,), rows in grouped.items():
        by_control[control_type] = {
            "primary_metric": metric,
            "primary": describe([row.get(metric) for row in rows]),
            "mean_term_recall": describe([row.get("mean_term_recall") for row in rows]),
            "all_terms_hit_rate": describe([row.get("all_terms_hit_rate") for row in rows]),
            "mean_reference_char_f1": describe([row.get("mean_reference_char_f1") for row in rows]),
            "mean_reference_chrf": describe([row.get("mean_reference_chrf") for row in rows]),
            "reference_contained_rate": describe([row.get("reference_contained_rate") for row in rows]),
            "num_examples": sorted({row.get("num_examples") for row in rows}),
        }
    gap = None
    if "real" in by_control and "random_target" in by_control:
        gap = by_control["real"]["primary"]["mean"] - by_control["random_target"]["primary"]["mean"]
    return {"primary_metric": metric, "by_control": by_control, "real_minus_random_gap": gap}


def summarize_geometry(geometry_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_control_extractor = {}
    for (control_type, extractor), rows in group_rows(geometry_rows, "control_type", "extractor").items():
        angle_values = [row.get("mean_delta_angle_degrees") for row in rows]
        cosine_values = [row.get("mean_delta_cosine") for row in rows]
        by_control_extractor[f"{control_type}/{extractor}"] = {
            "control_type": control_type,
            "extractor": extractor,
            "delta_angle_degrees": describe(angle_values),
            "delta_cosine": describe(cosine_values),
            "seeds_with_negative_delta_angle": sum(float(value) < 0 for value in numeric(angle_values)),
            "seeds_with_positive_delta_cosine": sum(float(value) > 0 for value in numeric(cosine_values)),
            "n_seeds": len(rows),
            "mean_negative_delta_angle_layers": describe([row.get("negative_delta_angle_layers") for row in rows]),
            "mean_positive_delta_cosine_layers": describe([row.get("positive_delta_cosine_layers") for row in rows]),
            "rows": rows,
        }

    paired = {}
    extractors = sorted({row.get("extractor") for row in geometry_rows})
    for extractor in extractors:
        real = {
            row.get("seed_index"): row
            for row in geometry_rows
            if row.get("control_type") == "real" and row.get("extractor") == extractor
        }
        random_target = {
            row.get("seed_index"): row
            for row in geometry_rows
            if row.get("control_type") == "random_target" and row.get("extractor") == extractor
        }
        seed_ids = sorted(set(real) & set(random_target))
        angle_diff = [
            float(real[seed]["mean_delta_angle_degrees"]) - float(random_target[seed]["mean_delta_angle_degrees"])
            for seed in seed_ids
        ]
        cosine_diff = [
            float(real[seed]["mean_delta_cosine"]) - float(random_target[seed]["mean_delta_cosine"])
            for seed in seed_ids
        ]
        paired[extractor] = {
            "n_paired_seeds": len(seed_ids),
            "seed_ids": seed_ids,
            "real_minus_random_delta_angle_degrees": describe(angle_diff),
            "real_minus_random_delta_cosine": describe(cosine_diff),
            "seeds_where_real_angle_decreased_more_than_random": sum(value < 0 for value in angle_diff),
            "seeds_where_real_cosine_increased_more_than_random": sum(value > 0 for value in cosine_diff),
        }

    return {"by_control_extractor": by_control_extractor, "paired_real_minus_random": paired}


def summarize_geometry_from_layers(detailed_rows: list[dict[str, Any]], layers: list[int]) -> dict[str, Any]:
    rows = []
    layer_set = set(layers)
    for (control_type, seed_index, extractor), group in group_rows(detailed_rows, "control_type", "seed_index", "extractor").items():
        subset = [row for row in group if row["layer"] in layer_set]
        if not subset:
            continue
        rows.append(
            {
                "control_type": control_type,
                "seed_index": seed_index,
                "extractor": extractor,
                "n_layers": len(subset),
                "mean_delta_angle_degrees": sum(row["delta_angle_degrees"] for row in subset) / len(subset),
                "mean_delta_cosine": sum(row["delta_cosine"] for row in subset) / len(subset),
                "negative_delta_angle_layers": sum(row["delta_angle_degrees"] < 0 for row in subset),
                "positive_delta_cosine_layers": sum(row["delta_cosine"] > 0 for row in subset),
            }
        )
    return summarize_geometry(rows)


def summarize_geometry_layer_sets(detailed_rows: list[dict[str, Any]], layer_sets: dict[str, list[int]]) -> dict[str, Any]:
    return {
        name: {
            "layers": layers,
            **summarize_geometry_from_layers(detailed_rows, layers),
        }
        for name, layers in layer_sets.items()
    }


def summarize_layers(detailed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    layer_rows = []
    for (control_type, extractor, layer), rows in sorted(group_rows(detailed_rows, "control_type", "extractor", "layer").items()):
        angle = describe([row["delta_angle_degrees"] for row in rows])
        cosine = describe([row["delta_cosine"] for row in rows])
        layer_rows.append(
            {
                "control_type": control_type,
                "extractor": extractor,
                "layer": layer,
                "n_seeds": len(rows),
                "mean_delta_angle_degrees": angle["mean"],
                "stdev_delta_angle_degrees": angle["stdev"],
                "mean_delta_cosine": cosine["mean"],
                "stdev_delta_cosine": cosine["stdev"],
                "seeds_with_negative_delta_angle": sum(row["delta_angle_degrees"] < 0 for row in rows),
                "seeds_with_positive_delta_cosine": sum(row["delta_cosine"] > 0 for row in rows),
            }
        )
    return layer_rows


def write_layer_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def verdict(report: dict[str, Any]) -> list[str]:
    lines = []
    behavior_gap = report["behavior"].get("real_minus_random_gap")
    if behavior_gap is not None:
        if behavior_gap > 0.1:
            lines.append(f"Behavior control passed: real LoRA beats random-target by {fmt(behavior_gap)} on the primary behavior metric.")
        else:
            lines.append(f"Behavior control is weak: real-minus-random primary behavior gap is only {fmt(behavior_gap)}.")

    paired = report["geometry"]["paired_real_minus_random"]
    linear = [name for name in ["mean_diff", "logistic"] if name in paired]
    supporting = []
    for extractor in linear:
        stats = paired[extractor]
        angle_mean = stats["real_minus_random_delta_angle_degrees"]["mean"]
        cosine_mean = stats["real_minus_random_delta_cosine"]["mean"]
        if angle_mean is not None and cosine_mean is not None and angle_mean < 0 and cosine_mean > 0:
            supporting.append(extractor)
    if supporting:
        lines.append("Linear geometry favors real LoRA over random-target for: " + ", ".join(supporting) + ".")
    else:
        lines.append("Linear geometry does not consistently favor real LoRA over random-target in the late-layer aggregate.")

    if "rfm" in paired:
        rfm_angle = paired["rfm"]["real_minus_random_delta_angle_degrees"]["mean"]
        if rfm_angle is not None and rfm_angle < 0:
            lines.append("RFM also favors real LoRA in the paired angle aggregate.")
        else:
            lines.append("RFM does not support a stable real-LoRA alignment story in the paired angle aggregate.")
    return lines


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def build_markdown(report: dict[str, Any], summary_dir: Path) -> str:
    lines = [
        "# Translation Geometry Final Report",
        "",
        f"Summary dir: `{summary_dir}`",
        "",
        "## Verdict",
        "",
    ]
    lines.extend(f"- {line}" for line in verdict(report))

    behavior = report["behavior"]
    lines.extend(["", "## Behavior", ""])
    behavior_rows = []
    for control in ["real", "random_target"]:
        if control not in behavior["by_control"]:
            continue
        stats = behavior["by_control"][control]
        behavior_rows.append(
            [
                control,
                stats["primary_metric"],
                fmt_ci(stats["primary"]),
                fmt_ci(stats["mean_term_recall"]),
                fmt_ci(stats["mean_reference_chrf"]),
                fmt_ci(stats["mean_reference_char_f1"]),
                ", ".join(str(v) for v in stats["num_examples"]),
            ]
        )
    lines.append(markdown_table(["control", "primary", "primary mean [95% CI]", "term recall", "chrF", "char F1", "n examples"], behavior_rows))
    lines.append(f"\nReal - random primary gap: `{fmt(behavior.get('real_minus_random_gap'))}`")

    lines.extend(["", "## Late-Layer Geometry", ""])
    geom_rows = []
    geom = report["geometry"]["by_control_extractor"]
    for key in sorted(geom):
        item = geom[key]
        geom_rows.append(
            [
                item["control_type"],
                item["extractor"],
                fmt_ci(item["delta_angle_degrees"]),
                fmt_ci(item["delta_cosine"], digits=4),
                f"{item['seeds_with_negative_delta_angle']}/{item['n_seeds']}",
                f"{item['seeds_with_positive_delta_cosine']}/{item['n_seeds']}",
            ]
        )
    lines.append(markdown_table(["control", "extractor", "delta angle", "delta cosine", "neg angle seeds", "pos cosine seeds"], geom_rows))

    lines.extend(["", "## Paired Real Minus Random", ""])
    paired_rows = []
    for extractor, item in sorted(report["geometry"]["paired_real_minus_random"].items()):
        paired_rows.append(
            [
                extractor,
                fmt_ci(item["real_minus_random_delta_angle_degrees"]),
                fmt_ci(item["real_minus_random_delta_cosine"], digits=4),
                f"{item['seeds_where_real_angle_decreased_more_than_random']}/{item['n_paired_seeds']}",
                f"{item['seeds_where_real_cosine_increased_more_than_random']}/{item['n_paired_seeds']}",
            ]
        )
    lines.append(markdown_table(["extractor", "real-random delta angle", "real-random delta cosine", "angle wins", "cosine wins"], paired_rows))

    layer_set_summaries = report.get("layer_set_geometry", {})
    if layer_set_summaries:
        lines.extend(["", "## Layer-Set Checks", ""])
        for name, layer_summary in layer_set_summaries.items():
            lines.extend(["", f"### {name} `{layer_summary['layers']}`", ""])
            rows = []
            for extractor, item in sorted(layer_summary["paired_real_minus_random"].items()):
                rows.append(
                    [
                        extractor,
                        fmt_ci(item["real_minus_random_delta_angle_degrees"]),
                        fmt_ci(item["real_minus_random_delta_cosine"], digits=4),
                        f"{item['seeds_where_real_angle_decreased_more_than_random']}/{item['n_paired_seeds']}",
                        f"{item['seeds_where_real_cosine_increased_more_than_random']}/{item['n_paired_seeds']}",
                    ]
                )
            lines.append(markdown_table(["extractor", "real-random delta angle", "real-random delta cosine", "angle wins", "cosine wins"], rows))

    lines.extend(["", "## Notes", ""])
    lines.append("- Negative `delta angle` means LoRA made `v_lang` and `v_trans` closer than base.")
    lines.append("- Positive `delta cosine` means the same thing in cosine form.")
    lines.append("- In the paired table, negative `real-random delta angle` means real LoRA aligned more than random-target LoRA.")
    lines.append("- Layer-set checks separate pre-output late layers from the final layer, which can behave differently near the output head.")
    lines.append("- With few seeds, 95% CIs are deliberately wide; use them as uncertainty flags, not as a final significance test.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    layer_sets = parse_layer_sets(args.layer_sets)
    summary_dir = Path(args.summary_dir) if args.summary_dir else latest_summary_dir(Path(args.summary_root))
    if not summary_dir.is_absolute():
        summary_dir = REPO_ROOT / summary_dir
    geometry_rows = read_csv_rows(summary_dir / "geometry_summary.csv")
    behavior_rows = read_csv_rows(summary_dir / "behavior_summary.csv")
    detailed_rows = load_detailed_layer_rows(geometry_rows, summary_dir)
    layer_rows = summarize_layers(detailed_rows)

    report = {
        "summary_dir": str(summary_dir),
        "geometry": summarize_geometry(geometry_rows),
        "layer_sets": layer_sets,
        "layer_set_geometry": summarize_geometry_layer_sets(detailed_rows, layer_sets),
        "behavior": summarize_behavior(behavior_rows),
        "layer_rows": layer_rows,
    }

    out_md = Path(args.out_md) if args.out_md else summary_dir / "final_report.md"
    out_json = Path(args.out_json) if args.out_json else summary_dir / "final_report.json"
    out_layer_csv = Path(args.out_layer_csv) if args.out_layer_csv else summary_dir / "layer_summary.csv"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_layer_csv.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text(build_markdown(report, summary_dir), encoding="utf-8")
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_layer_csv(out_layer_csv, layer_rows)

    print(json.dumps({"summary_dir": str(summary_dir), "report_md": str(out_md), "report_json": str(out_json), "layer_csv": str(out_layer_csv)}, indent=2))


if __name__ == "__main__":
    main()
