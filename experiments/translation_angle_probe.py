"""Measure language-vs-translation direction angles on a small EN-ZH set."""

from __future__ import annotations

import argparse
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

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - peft is optional for base-model probes.
    PeftModel = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small EN-ZH translation angle probe.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lora-path", default=None, help="Optional PEFT LoRA adapter path.")
    parser.add_argument("--model-state", default=None, help="Label written to metrics, such as base or translation_lora.")
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data" / "translation_mini" / "en_zh_mini.jsonl"))
    parser.add_argument("--splits", nargs="*", default=None, help="Optional split names to include in legacy single-split mode.")
    parser.add_argument("--fit-split", default=None, help="Split used to fit concept vectors.")
    parser.add_argument("--test-split", default=None, help="Independent split used to estimate/report concept-vector angles.")
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="0-based transformer layer indices.")
    parser.add_argument("--extractors", nargs="*", default=["mean_diff", "logistic"], choices=["mean_diff", "logistic", "mlp_agop"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    parser.add_argument("--bootstrap-mlp-agop", action="store_true", help="Also bootstrap MLP-AGOP; this is much slower.")
    parser.add_argument("--mlp-agop-steps", type=int, default=120)
    parser.add_argument("--mlp-agop-hidden", type=int, default=128)
    parser.add_argument("--mlp-agop-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-agop-device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "translation_angle"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
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
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def read_pairs(path: str) -> list[dict[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                pairs.append(json.loads(line))
    if not pairs:
        raise ValueError(f"No translation pairs found in {path}")
    return pairs


def filter_pairs(pairs: list[dict[str, str]], splits: list[str] | None) -> list[dict[str, str]]:
    if not splits:
        return pairs
    split_set = set(splits)
    filtered = [pair for pair in pairs if pair.get("split") in split_set]
    if not filtered:
        raise ValueError(f"No examples found for splits: {sorted(split_set)}")
    return filtered


def default_layers(num_layers: int) -> list[int]:
    return sorted({max(0, min(num_layers - 1, round(num_layers * frac) - 1)) for frac in (0.25, 0.50, 0.65, 0.80)})


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + eps)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(cosine(a, b), -1.0, 1.0))))


def iter_batches(size: int, batch_size: int):
    for start in range(0, size, batch_size):
        yield slice(start, min(start + batch_size, size))


def build_language_dataset(pairs: list[dict[str, str]]) -> tuple[list[str], np.ndarray, list[dict]]:
    texts, labels, metadata = [], [], []
    for pair in pairs:
        texts.append(pair["en"])
        labels.append(0)
        metadata.append({"id": pair["id"], "split": pair.get("split"), "task": "language", "language": "en", "label": 0, "text": pair["en"]})
        texts.append(pair["zh"])
        labels.append(1)
        metadata.append({"id": pair["id"], "split": pair.get("split"), "task": "language", "language": "zh", "label": 1, "text": pair["zh"]})
    return texts, np.asarray(labels, dtype=np.int64), metadata


def build_translation_activation_dataset(pairs: list[dict[str, str]]) -> tuple[list[str], np.ndarray, list[dict]]:
    texts, labels, metadata = [], [], []
    for pair in pairs:
        active = (
            "Task: translate the English sentence into Chinese.\n"
            f"Sentence: {pair['en']}\n"
            "Answer:"
        )
        inactive = (
            "Task: repeat the English sentence without translating it.\n"
            f"Sentence: {pair['en']}\n"
            "Answer:"
        )
        texts.append(active)
        labels.append(1)
        metadata.append({"id": pair["id"], "split": pair.get("split"), "task": "translation_activation", "control": "translate", "label": 1, "text": active})
        texts.append(inactive)
        labels.append(0)
        metadata.append({"id": pair["id"], "split": pair.get("split"), "task": "translation_activation", "control": "repeat", "label": 0, "text": inactive})
    return texts, np.asarray(labels, dtype=np.int64), metadata


def extract_last_token_activations(
    model,
    tokenizer,
    texts: list[str],
    layers: list[int],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    activations = {layer: [] for layer in layers}
    model.eval()
    with torch.no_grad():
        for batch_slice in iter_batches(len(texts), batch_size):
            encoded = tokenizer(
                texts[batch_slice],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True)
            last_token_idx = encoded["attention_mask"].sum(dim=1) - 1
            row_idx = torch.arange(last_token_idx.shape[0], device=device)
            for layer in layers:
                hidden = outputs.hidden_states[layer + 1]
                acts = hidden[row_idx, last_token_idx].detach().cpu().float().numpy()
                activations[layer].append(acts)
    return {layer: np.concatenate(parts, axis=0) for layer, parts in activations.items()}


def fit_mean_diff(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    vector = normalize(x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0))
    scores = x @ vector
    auc = float(roc_auc_score(y, scores))
    return vector, auc


def fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
    model.fit(x, y)
    vector = normalize(model.coef_.reshape(-1))
    scores = model.decision_function(x)
    auc = float(roc_auc_score(y, scores))
    return vector, auc


class MLPProbe(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_mlp_agop(
    x: np.ndarray,
    y: np.ndarray,
    steps: int,
    hidden_dim: int,
    lr: float,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Approximate an RFM/AGOP concept vector with gradients of a small MLP probe."""
    x_tensor = torch.from_numpy(x).float().to(device)
    y_tensor = torch.from_numpy(y.astype(np.float32)).to(device)
    model = MLPProbe(input_dim=x.shape[1], hidden_dim=min(hidden_dim, x.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    x_grad = x_tensor.detach().clone().requires_grad_(True)
    logits = model(x_grad)
    grads = torch.autograd.grad(logits.sum(), x_grad)[0].detach().cpu()
    grads = grads - grads.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(grads, full_matrices=False)
    vector = normalize(vh[0].numpy().astype(np.float32))
    auc = projection_auc(x, y, vector)
    return vector, auc


def make_fitters(args: argparse.Namespace, device: torch.device):
    agop_device = torch.device(args.mlp_agop_device)
    if agop_device.type == "mps" and not torch.backends.mps.is_available():
        agop_device = torch.device("cpu")
    if agop_device.type == "cuda" and not torch.cuda.is_available():
        agop_device = torch.device("cpu")
    torch.manual_seed(args.seed)
    return {
        "mean_diff": fit_mean_diff,
        "logistic": fit_logistic,
        "mlp_agop": lambda x, y: fit_mlp_agop(
            x,
            y,
            steps=args.mlp_agop_steps,
            hidden_dim=args.mlp_agop_hidden,
            lr=args.mlp_agop_lr,
            device=agop_device,
        ),
    }


def projection_auc(x: np.ndarray, y: np.ndarray, vector: np.ndarray) -> float:
    return float(roc_auc_score(y, x @ normalize(vector)))


def compute_angles(
    lang_acts: dict[int, np.ndarray],
    lang_labels: np.ndarray,
    trans_acts: dict[int, np.ndarray],
    trans_labels: np.ndarray,
    extractors: list[str],
    split_name: str,
    eval_lang_acts: dict[int, np.ndarray] | None = None,
    eval_lang_labels: np.ndarray | None = None,
    eval_trans_acts: dict[int, np.ndarray] | None = None,
    eval_trans_labels: np.ndarray | None = None,
    fitters: dict | None = None,
) -> list[dict]:
    rows = []
    for layer in lang_acts:
        for extractor_name in extractors:
            fitter = fitters[extractor_name]
            lang_vec, lang_auc = fitter(lang_acts[layer], lang_labels)
            trans_vec, trans_auc = fitter(trans_acts[layer], trans_labels)
            row = {
                "extractor": extractor_name,
                "layer": layer,
                "split": split_name,
                "cosine": cosine(lang_vec, trans_vec),
                "angle_degrees": angle_degrees(lang_vec, trans_vec),
                "language_auc": lang_auc,
                "translation_auc": trans_auc,
            }
            if eval_lang_acts is not None and eval_trans_acts is not None:
                row["language_auc_eval"] = projection_auc(eval_lang_acts[layer], eval_lang_labels, lang_vec)
                row["translation_auc_eval"] = projection_auc(eval_trans_acts[layer], eval_trans_labels, trans_vec)
            rows.append(row)
    return rows


def pair_row_indices(pair_indices: np.ndarray) -> np.ndarray:
    rows = np.empty(len(pair_indices) * 2, dtype=np.int64)
    rows[0::2] = pair_indices * 2
    rows[1::2] = pair_indices * 2 + 1
    return rows


def subset_activations(acts: dict[int, np.ndarray], rows: np.ndarray) -> dict[int, np.ndarray]:
    return {layer: values[rows] for layer, values in acts.items()}


def bootstrap_rows(
    rows: list[dict],
    lang_acts: dict[int, np.ndarray],
    lang_labels: np.ndarray,
    trans_acts: dict[int, np.ndarray],
    trans_labels: np.ndarray,
    extractors: list[str],
    fitters: dict,
    bootstrap_mlp_agop: bool,
    n_pairs: int,
    n_boot: int,
    seed: int,
) -> list[dict]:
    if n_boot <= 0:
        return rows
    rng = np.random.default_rng(seed)
    bootstrap_extractors = [name for name in extractors if name != "mlp_agop" or bootstrap_mlp_agop]
    values: dict[tuple[str, int], dict[str, list[float]]] = {
        (row["extractor"], row["layer"]): {"cosine": [], "angle_degrees": []}
        for row in rows
        if row["extractor"] in bootstrap_extractors
    }
    if not bootstrap_extractors:
        return rows
    for _ in range(n_boot):
        sampled_pairs = rng.integers(0, n_pairs, size=n_pairs)
        sampled_rows = pair_row_indices(sampled_pairs)
        boot_lang_acts = subset_activations(lang_acts, sampled_rows)
        boot_trans_acts = subset_activations(trans_acts, sampled_rows)
        boot_lang_labels = lang_labels[sampled_rows]
        boot_trans_labels = trans_labels[sampled_rows]
        for row in compute_angles(
            boot_lang_acts,
            boot_lang_labels,
            boot_trans_acts,
            boot_trans_labels,
            bootstrap_extractors,
            split_name="bootstrap",
            fitters=fitters,
        ):
            key = (row["extractor"], row["layer"])
            values[key]["cosine"].append(row["cosine"])
            values[key]["angle_degrees"].append(row["angle_degrees"])

    for row in rows:
        key = (row["extractor"], row["layer"])
        if key not in values:
            continue
        for metric in ["cosine", "angle_degrees"]:
            samples = np.asarray(values[key][metric])
            row[f"{metric}_boot_mean"] = float(samples.mean())
            row[f"{metric}_ci_low"] = float(np.percentile(samples, 2.5))
            row[f"{metric}_ci_high"] = float(np.percentile(samples, 97.5))
    return rows


def write_plots(rows: list[dict], output_dir: Path) -> None:
    extractors = sorted({row["extractor"] for row in rows})
    for metric, ylabel, filename in [
        ("angle_degrees", "Angle (degrees)", "angle_by_layer.png"),
        ("cosine", "Cosine similarity", "cosine_by_layer.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for extractor in extractors:
            subset = sorted([row for row in rows if row["extractor"] == extractor], key=lambda row: row["layer"])
            ax.plot(
                [row["layer"] for row in subset],
                [row[metric] for row in subset],
                marker="o",
                linewidth=2,
                label=extractor,
            )
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def get_hidden_size(config) -> int:
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "n_embd", None)
    if hidden_size is None:
        raise AttributeError("Could not find hidden size on model config.")
    return int(hidden_size)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)

    all_pairs = read_pairs(args.data_path)
    if args.fit_split or args.test_split:
        if not args.fit_split or not args.test_split:
            raise ValueError("--fit-split and --test-split must be provided together.")
        fit_pairs = filter_pairs(all_pairs, [args.fit_split])
        test_pairs = filter_pairs(all_pairs, [args.test_split])
    else:
        fit_pairs = filter_pairs(all_pairs, args.splits)
        test_pairs = fit_pairs
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    if args.lora_path:
        if PeftModel is None:
            raise ImportError("peft is required to load --lora-path")
        model = PeftModel.from_pretrained(model, args.lora_path).to(device)

    num_layers = int(getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", 0)))
    if num_layers <= 0:
        raise AttributeError("Could not infer number of hidden layers.")
    layers = args.layers if args.layers else default_layers(num_layers)
    fitters = make_fitters(args, device)

    fit_lang_texts, fit_lang_labels, fit_lang_meta = build_language_dataset(fit_pairs)
    fit_trans_texts, fit_trans_labels, fit_trans_meta = build_translation_activation_dataset(fit_pairs)
    test_lang_texts, test_lang_labels, test_lang_meta = build_language_dataset(test_pairs)
    test_trans_texts, test_trans_labels, test_trans_meta = build_translation_activation_dataset(test_pairs)

    fit_lang_acts = extract_last_token_activations(model, tokenizer, fit_lang_texts, layers, args.batch_size, args.max_length, device)
    fit_trans_acts = extract_last_token_activations(model, tokenizer, fit_trans_texts, layers, args.batch_size, args.max_length, device)
    if fit_pairs is test_pairs:
        test_lang_acts, test_trans_acts = fit_lang_acts, fit_trans_acts
        test_lang_labels, test_trans_labels = fit_lang_labels, fit_trans_labels
    else:
        test_lang_acts = extract_last_token_activations(model, tokenizer, test_lang_texts, layers, args.batch_size, args.max_length, device)
        test_trans_acts = extract_last_token_activations(model, tokenizer, test_trans_texts, layers, args.batch_size, args.max_length, device)
    fit_rows = compute_angles(
        fit_lang_acts,
        fit_lang_labels,
        fit_trans_acts,
        fit_trans_labels,
        args.extractors,
        split_name=args.fit_split or "fit",
        eval_lang_acts=test_lang_acts,
        eval_lang_labels=test_lang_labels,
        eval_trans_acts=test_trans_acts,
        eval_trans_labels=test_trans_labels,
        fitters=fitters,
    )
    rows = compute_angles(
        test_lang_acts,
        test_lang_labels,
        test_trans_acts,
        test_trans_labels,
        args.extractors,
        split_name=args.test_split or "test",
        fitters=fitters,
    )
    rows = bootstrap_rows(
        rows,
        test_lang_acts,
        test_lang_labels,
        test_trans_acts,
        test_trans_labels,
        args.extractors,
        fitters,
        args.bootstrap_mlp_agop,
        n_pairs=len(test_pairs),
        n_boot=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )

    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    np.savez_compressed(output_dir / "fit_language_activations.npz", **{f"layer_{k}": v for k, v in fit_lang_acts.items()})
    np.savez_compressed(output_dir / "fit_translation_activations.npz", **{f"layer_{k}": v for k, v in fit_trans_acts.items()})
    np.savez_compressed(output_dir / "test_language_activations.npz", **{f"layer_{k}": v for k, v in test_lang_acts.items()})
    np.savez_compressed(output_dir / "test_translation_activations.npz", **{f"layer_{k}": v for k, v in test_trans_acts.items()})
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for item in fit_lang_meta + fit_trans_meta + test_lang_meta + test_trans_meta:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (output_dir / "angles.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "fit_angles.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in fit_rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fit_rows)
    write_plots(rows, output_dir)

    metrics = {
        "run_id": run_id,
        "model_id": args.model_id,
        "model_state": args.model_state or ("translation_lora" if args.lora_path else "base"),
        "lora_path": args.lora_path,
        "data_path": args.data_path,
        "splits": args.splits,
        "fit_split": args.fit_split,
        "test_split": args.test_split,
        "num_fit_pairs": len(fit_pairs),
        "num_test_pairs": len(test_pairs),
        "bootstrap_samples": args.bootstrap_samples,
        "extractors": args.extractors,
        "device": str(device),
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "num_pairs": len(test_pairs),
        "num_layers": num_layers,
        "hidden_size": get_hidden_size(model.config),
        "layers": layers,
        "fit_rows": fit_rows,
        "rows": rows,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(json.dumps({"output_dir": str(output_dir), **metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
