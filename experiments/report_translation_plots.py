"""Plot layer-wise translation geometry summaries."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


from report_translation_results import (
    DEFAULT_SUMMARY_ROOT,
    REPO_ROOT,
    describe,
    latest_summary_dir,
    load_detailed_layer_rows,
    read_csv_rows,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXTRACTOR_COLORS = {
    "mean_diff": "#3b82f6",
    "logistic": "#f97316",
    "rfm": "#16a34a",
    "mlp_agop": "#9333ea",
}

CONTROL_STYLES = {
    "real": "-",
    "random_target": "--",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create layer-wise translation geometry plots.")
    parser.add_argument("--summary-dir", default=None, help="Run directory under artifacts/translation_rigorous_summary. Defaults to latest.")
    parser.add_argument("--summary-root", default=str(DEFAULT_SUMMARY_ROOT))
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def group_rows(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def sem(values: list[float]) -> float:
    stats = describe(values)
    return float(stats["sem"] or 0.0)


def style_axes(ax, ylabel: str) -> None:
    ax.axhline(0, color="#111827", linewidth=1, alpha=0.45)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_control_curves(rows: list[dict[str, Any]], metric: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    grouped = group_rows(rows, "control_type", "extractor", "layer")
    controls = sorted({row["control_type"] for row in rows})
    extractors = sorted({row["extractor"] for row in rows})
    for control in controls:
        for extractor in extractors:
            points = []
            for (row_control, row_extractor, layer), group in grouped.items():
                if row_control == control and row_extractor == extractor:
                    values = [float(row[metric]) for row in group]
                    points.append((int(layer), sum(values) / len(values), sem(values)))
            if not points:
                continue
            points.sort()
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            errs = [point[2] for point in points]
            color = EXTRACTOR_COLORS.get(extractor)
            label = f"{control} / {extractor}"
            ax.plot(xs, ys, CONTROL_STYLES.get(control, "-"), marker="o", markersize=3.5, linewidth=1.8, color=color, label=label)
            ax.fill_between(xs, [y - e for y, e in zip(ys, errs)], [y + e for y, e in zip(ys, errs)], color=color, alpha=0.10)
    style_axes(ax, ylabel)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["control_type"], row["seed_index"], row["extractor"], row["layer"]): row
        for row in rows
    }
    output = []
    seed_ids = sorted({row["seed_index"] for row in rows})
    extractors = sorted({row["extractor"] for row in rows})
    layers = sorted({row["layer"] for row in rows})
    for seed in seed_ids:
        for extractor in extractors:
            for layer in layers:
                real = by_key.get(("real", seed, extractor, layer))
                random_target = by_key.get(("random_target", seed, extractor, layer))
                if not real or not random_target:
                    continue
                output.append(
                    {
                        "seed_index": seed,
                        "extractor": extractor,
                        "layer": layer,
                        "delta_angle_degrees": real["delta_angle_degrees"] - random_target["delta_angle_degrees"],
                        "delta_cosine": real["delta_cosine"] - random_target["delta_cosine"],
                    }
                )
    return output


def plot_paired_curves(rows: list[dict[str, Any]], metric: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    grouped = group_rows(rows, "extractor", "layer")
    for extractor in sorted({row["extractor"] for row in rows}):
        points = []
        for (row_extractor, layer), group in grouped.items():
            if row_extractor != extractor:
                continue
            values = [float(row[metric]) for row in group]
            points.append((int(layer), sum(values) / len(values), sem(values)))
        if not points:
            continue
        points.sort()
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        errs = [point[2] for point in points]
        color = EXTRACTOR_COLORS.get(extractor)
        ax.plot(xs, ys, marker="o", markersize=3.8, linewidth=2.0, color=color, label=extractor)
        ax.fill_between(xs, [y - e for y, e in zip(ys, errs)], [y + e for y, e in zip(ys, errs)], color=color, alpha=0.12)
    style_axes(ax, ylabel)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir) if args.summary_dir else latest_summary_dir(Path(args.summary_root))
    if not summary_dir.is_absolute():
        summary_dir = REPO_ROOT / summary_dir
    out_dir = Path(args.out_dir) if args.out_dir else summary_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry_rows = read_csv_rows(summary_dir / "geometry_summary.csv")
    detailed_rows = load_detailed_layer_rows(geometry_rows, summary_dir)
    if not detailed_rows:
        raise FileNotFoundError(f"No detailed layer rows could be loaded for {summary_dir}")
    paired = paired_rows(detailed_rows)

    plot_control_curves(detailed_rows, "delta_angle_degrees", "Delta angle, LoRA - base", out_dir / "layer_delta_angle_by_control.png")
    plot_control_curves(detailed_rows, "delta_cosine", "Delta cosine, LoRA - base", out_dir / "layer_delta_cosine_by_control.png")
    plot_paired_curves(paired, "delta_angle_degrees", "Real - random delta angle", out_dir / "layer_paired_delta_angle.png")
    plot_paired_curves(paired, "delta_cosine", "Real - random delta cosine", out_dir / "layer_paired_delta_cosine.png")

    print(
        {
            "summary_dir": str(summary_dir),
            "plots": [
                str(out_dir / "layer_delta_angle_by_control.png"),
                str(out_dir / "layer_delta_cosine_by_control.png"),
                str(out_dir / "layer_paired_delta_angle.png"),
                str(out_dir / "layer_paired_delta_cosine.png"),
            ],
        }
    )


if __name__ == "__main__":
    main()
