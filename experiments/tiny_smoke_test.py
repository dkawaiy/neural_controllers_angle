"""Run a tiny, CPU-friendly neural-controller smoke test.

The goal is not to prove the research hypothesis. It verifies that a
decoder-only model can be loaded, hidden states can be extracted from
selected layers, and simple concept directions can be fit and saved.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".hf_cache" / "datasets"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


SCIENCE_PROMPTS = [
    "The astronomy note describes a galaxy, a telescope, and a distant star.",
    "The biology paragraph explains cells, proteins, and a microscope.",
    "The physics memo studies force, energy, and orbital motion.",
    "The chemistry summary mentions atoms, molecules, and a laboratory.",
    "The geology report describes minerals, fossils, and tectonic plates.",
    "The climate note discusses carbon, oceans, and temperature records.",
    "The mathematics proof uses vectors, matrices, and a theorem.",
    "The neuroscience article studies neurons, memory, and brain signals.",
    "The engineering brief tests circuits, sensors, and a prototype.",
    "The medical note describes a vaccine, immunity, and clinical data.",
    "The space mission report tracks a satellite and a planetary orbit.",
    "The genetics lesson explains DNA, inheritance, and mutations.",
]

COOKING_PROMPTS = [
    "The cooking note describes a soup, a pan, and fresh herbs.",
    "The baking paragraph explains flour, butter, and a warm oven.",
    "The kitchen memo studies sauce, salt, and roasted vegetables.",
    "The recipe summary mentions noodles, garlic, and olive oil.",
    "The dinner report describes rice, beans, and a simmering pot.",
    "The cafe note discusses coffee, sugar, and breakfast pastries.",
    "The dessert guide uses chocolate, cream, and a mixing bowl.",
    "The restaurant article studies menus, service, and seasonal dishes.",
    "The lunch brief tests bread, cheese, and a grilled sandwich.",
    "The meal note describes soup stock, onions, and a sharp knife.",
    "The market report tracks fruit, fish, and a basket of vegetables.",
    "The pantry lesson explains spices, lentils, and dried pasta.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny model smoke test for neural-controller infrastructure.")
    parser.add_argument("--model-id", default=os.environ.get("SMOKE_MODEL_ID", "EleutherAI/pythia-14m"))
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="0-based transformer layer indices.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "smoke"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + eps)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))


def build_dataset() -> tuple[list[str], np.ndarray]:
    prompts = SCIENCE_PROMPTS + COOKING_PROMPTS
    labels = np.array([1] * len(SCIENCE_PROMPTS) + [0] * len(COOKING_PROMPTS), dtype=np.int64)
    return prompts, labels


def iter_batches(size: int, batch_size: int):
    for start in range(0, size, batch_size):
        yield slice(start, min(start + batch_size, size))


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
                layer_acts = hidden[row_idx, last_token_idx].detach().cpu().float().numpy()
                activations[layer].append(layer_acts)
    return {layer: np.concatenate(parts, axis=0) for layer, parts in activations.items()}


def fit_direction_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mean_vector = normalize(x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0))
    mean_scores = x @ mean_vector
    logistic = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
    logistic.fit(x, y)
    logistic_vector = normalize(logistic.coef_.reshape(-1))
    logistic_scores = logistic.decision_function(x)
    return {
        "mean_diff_auc": float(roc_auc_score(y, mean_scores)),
        "logistic_auc": float(roc_auc_score(y, logistic_scores)),
        "mean_logistic_cosine": cosine(mean_vector, logistic_vector),
        "mean_vector_norm": float(np.linalg.norm(mean_vector)),
        "logistic_vector_norm": float(np.linalg.norm(logistic_vector)),
    }


def generate_probe_text(model, tokenizer, device: torch.device, max_new_tokens: int) -> str:
    prompt = "Science note:"
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


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

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = getattr(model.config, "num_hidden_layers", None) or getattr(model.config, "n_layer")
    default_layers = sorted(set([0, num_layers // 2, num_layers - 1]))
    layers = args.layers if args.layers else default_layers
    bad_layers = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if bad_layers:
        raise ValueError(f"Layer indices out of range for {num_layers} layers: {bad_layers}")

    texts, labels = build_dataset()
    activations = extract_last_token_activations(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        layers=layers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    layer_metrics = {str(layer): fit_direction_metrics(x, labels) for layer, x in activations.items()}
    generated_text = generate_probe_text(model, tokenizer, device, args.max_new_tokens)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    np.savez_compressed(output_dir / "activations.npz", **{f"layer_{k}": v for k, v in activations.items()})
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for idx, (text, label) in enumerate(zip(texts, labels.tolist())):
            handle.write(json.dumps({"sample_id": idx, "text": text, "label": label}) + "\n")

    metrics = {
        "run_id": run_id,
        "model_id": args.model_id,
        "device": str(device),
        "torch_version": torch.__version__,
        "num_layers": int(num_layers),
        "hidden_size": get_hidden_size(model.config),
        "layers": layers,
        "num_samples": len(texts),
        "layer_metrics": layer_metrics,
        "generated_text": generated_text,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(json.dumps({"output_dir": str(output_dir), **metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
