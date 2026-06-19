"""Non-probing representational geometry analyses for acceptance datasets.

This script measures geometry directly in hidden-state spaces. It never fits a
classifier or supervised probe: labels are used only to split points into the
two acceptability classes for descriptive geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

try:
    from skdim.id import ESS
except ImportError:  # pragma: no cover - optional estimator.
    ESS = None

try:
    import torch
except ImportError:  # pragma: no cover - metric-only use can still work.
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - metric-only use can still work.
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from utils import get_last_token_embeddings
except Exception:  # utils.py imports h5py; keep this script usable without it.
    if torch is None:
        def get_last_token_embeddings(*args, **kwargs):
            raise ImportError("torch is required for representation extraction.")
    else:
        from torch.utils.data import DataLoader

        @torch.inference_mode()
        def get_last_token_embeddings(
            model,
            texts,
            tokenizer,
            device,
            batch_size: int = 256,
            include_embedding_matrix: bool = True,
        ):
            model.eval()
            all_embs = []

            def collate(batch_texts):
                return tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )

            loader = DataLoader(texts, batch_size=batch_size, shuffle=False, collate_fn=collate)
            for batch in tqdm(loader, desc="Extracting representations", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states
                if not include_embedding_matrix:
                    hidden_states = hidden_states[1:]
                hidden_states = torch.stack(hidden_states, dim=0)

                last_idx = attention_mask.sum(dim=1) - 1
                batch_arange = torch.arange(input_ids.size(0), device=device)
                sent_emb = hidden_states[:, batch_arange, last_idx, :]
                sent_emb = sent_emb.permute(1, 0, 2).contiguous()
                all_embs.append(sent_emb.cpu())

            return torch.cat(all_embs, dim=0)


sns.set_theme(style="whitegrid", font_scale=0.9)

EPS = 1e-12
PAIRWISE_WITHIN_MAX = 2000
PAIRWISE_BETWEEN_MAX_PER_CLASS = 1000
SILHOUETTE_MAX = 2000
MMD_MAX_PER_CLASS = 600
ID_MAX_SAMPLES = 5000
ESS_MAX_SAMPLES = 60
ESS_N_NEIGHBORS = 10
PCA2D_MAX_SAMPLES = 1500
KNN_NEIGHBORS = 10
KNN_METRIC = "cosine"
CLASS_SUBSET_LABELS = {
    "class_0": "ungrammatical",
    "class_1": "grammatical",
}
CLASS_SUBSET_STYLES = {
    "class_0": {"linestyle": "--", "marker": "s", "alpha": 0.85},
    "class_1": {"linestyle": "-", "marker": "o", "alpha": 0.95},
}

CLASS_COLUMNS = [
    "split",
    "dataset",
    "fold",
    "model",
    "checkpoint",
    "checkpoint_step",
    "layer",
    "layer_depth",
    "n_total",
    "n_class_0",
    "n_class_1",
    "centroid_euclidean",
    "centroid_euclidean_class_0",
    "centroid_euclidean_class_1",
    "centroid_cosine_distance",
    "within_class_0_mean_cosine_distance",
    "within_class_1_mean_cosine_distance",
    "pooled_within_mean_cosine_distance",
    "between_class_mean_cosine_distance",
    "separation_ratio",
    "fisher_ratio",
    "fisher_ratio_class_0",
    "fisher_ratio_class_1",
    "silhouette_cosine",
    "mmd_rbf",
    "is_aggregate",
    "n_folds",
]

ID_COLUMNS = [
    "split",
    "dataset",
    "fold",
    "model",
    "checkpoint",
    "checkpoint_step",
    "layer",
    "layer_depth",
    "subset",
    "n_samples",
    "n_samples_used",
    "ambient_dim",
    "pca_rank_90",
    "pca_rank_95",
    "pca_rank_99",
    "effective_rank",
    "ess_id",
    "twonn_id",
    "mle_id_k5",
    "mle_id_k10",
    "mle_id_k20",
    "is_aggregate",
    "n_folds",
]

CKA_COLUMNS = [
    "split",
    "dataset",
    "fold",
    "checkpoint_a",
    "checkpoint_b",
    "checkpoint_step_a",
    "checkpoint_step_b",
    "model_a",
    "model_b",
    "layer_a",
    "layer_b",
    "layer_depth_a",
    "layer_depth_b",
    "n_aligned",
    "cka",
    "is_aggregate",
    "n_folds",
]

DRIFT_COLUMNS = [
    "split",
    "dataset",
    "fold",
    "model",
    "metric",
    "checkpoint_a",
    "checkpoint_b",
    "checkpoint_step_a",
    "checkpoint_step_b",
    "reference_checkpoint",
    "layer",
    "layer_depth",
    "label",
    "n_aligned",
    "cka",
    "centroid_drift_euclidean",
    "is_aggregate",
    "n_folds",
]

PCA2D_COLUMNS = [
    "split",
    "dataset",
    "source_datasets",
    "model",
    "checkpoint",
    "checkpoint_step",
    "layer",
    "layer_depth",
    "n_total",
    "n_plotted",
    "pc1_variance_ratio",
    "pc2_variance_ratio",
    "pca2d_variance_ratio",
    "is_aggregate",
    "n_folds",
]

PREFERENCE_DIRECTION_COLUMNS = [
    "split",
    "dataset",
    "source_datasets",
    "model",
    "checkpoint",
    "checkpoint_step",
    "layer",
    "layer_depth",
    "n_total",
    "n_class_0",
    "n_class_1",
    "direction_norm",
    "score_mean_class_0",
    "score_mean_class_1",
    "score_gap",
    "score_std_class_0",
    "score_std_class_1",
    "score_pooled_std",
    "score_cohens_d",
    "is_aggregate",
    "n_folds",
]

PREFERENCE_MODEL_AGREEMENT_COLUMNS = [
    "split",
    "dataset",
    "source_datasets",
    "checkpoint",
    "checkpoint_step",
    "model_a",
    "model_b",
    "layer_a",
    "layer_b",
    "layer_depth_a",
    "layer_depth_b",
    "n_aligned",
    "score_pearson",
    "score_centered_cosine",
    "is_aggregate",
    "n_folds",
]

PREFERENCE_TASK_AGREEMENT_COLUMNS = [
    "split",
    "model",
    "checkpoint",
    "checkpoint_step",
    "dataset_a",
    "dataset_b",
    "source_datasets_a",
    "source_datasets_b",
    "layer",
    "layer_depth",
    "direction_cosine",
    "direction_abs_cosine",
    "n_total_a",
    "n_total_b",
    "is_aggregate_a",
    "is_aggregate_b",
    "n_folds_a",
    "n_folds_b",
]

KNN_AGREEMENT_COLUMNS = [
    "split",
    "dataset",
    "fold",
    "checkpoint",
    "checkpoint_step",
    "model_a",
    "model_b",
    "layer_a",
    "layer_b",
    "layer_depth",
    "layer_depth_a",
    "layer_depth_b",
    "class_subset",
    "n_aligned",
    "n_items",
    "k",
    "k_effective",
    "distance_metric",
    "mean_jaccard",
    "std_jaccard",
    "mean_overlap_count",
    "mean_overlap_fraction",
    "random_overlap_fraction",
    "normalized_overlap_fraction",
    "is_aggregate",
    "n_folds",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    split: str = "test"


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path
    step: float


def warn(message: str) -> None:
    warnings.warn(message, stacklevel=2)


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def short_model_name(model_name: object) -> str:
    """Match the compact architecture labels used in the probing plots."""
    name = str(model_name)
    if name.startswith("gpt_"):
        name = name[len("gpt_") :]
    return re.sub(r"_\d+s$", "", name)


def checkpoint_plot_label(checkpoint: object) -> str:
    checkpoint = str(checkpoint)
    if checkpoint == "final":
        return "ckfinal"
    step = checkpoint_step(checkpoint)
    if np.isfinite(step):
        return f"ck{int(step)}"
    return safe_name(checkpoint)


def drop_temporal_fold_rows(df: pd.DataFrame, dataset_col: str = "dataset") -> pd.DataFrame:
    if df.empty or dataset_col not in df.columns:
        return df
    mask = df[dataset_col].astype(str).str.startswith("temporal_concord_fold_")
    return df[~mask].copy()


def style_plain_ax(ax, xlabel: str, ylabel: str) -> None:
    ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def stable_seed(*parts: object, base_seed: int) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) % (2**32 - 1)


def sort_names_with_architecture_hint(names: Iterable[str]) -> List[str]:
    def key(name: str) -> Tuple[int, int, str]:
        layers = re.search(r"(\d+)l", name)
        heads = re.search(r"(\d+)h", name)
        layer_key = int(layers.group(1)) if layers else 10**9
        head_key = int(heads.group(1)) if heads else 10**9
        return layer_key, head_key, name

    return sorted(names, key=key)


def normalize_checkpoint_label(value: str) -> str:
    value = str(value)
    if value.lower() == "final":
        return "final"
    if value.isdigit():
        return f"checkpoint-{value}"
    return value


def checkpoint_step(label: str) -> float:
    if label == "final":
        return math.nan
    match = re.search(r"checkpoint-(\d+)", label)
    return float(match.group(1)) if match else math.nan


def checkpoint_sort_key(label: str) -> Tuple[int, float, str]:
    if label == "final":
        return 1, math.inf, label
    step = checkpoint_step(label)
    return 0, step if np.isfinite(step) else math.inf, label


def layer_depth(layer: int, num_layers: int) -> float:
    return float(layer / (num_layers - 1)) if num_layers > 1 else 0.0


def parse_args() -> argparse.Namespace:
    default_device = "cpu"
    if torch is not None and torch.cuda.is_available():
        default_device = "cuda"

    parser = argparse.ArgumentParser(
        "Non-probing representation geometry over models, checkpoints, datasets, and layers."
    )
    parser.add_argument("--models_dir", type=str, default="models/pretrained")
    parser.add_argument("--tokenizer_path", type=str, default="models/tokenizer")
    parser.add_argument("--train_dir", type=str, default="data/acceptance_datasets/train")
    parser.add_argument("--test_dir", type=str, default="data/acceptance_datasets/test")
    parser.add_argument("--representations_dir", type=str, default="data/representations_geometry")
    parser.add_argument("--results_dir", type=str, default="data/results")
    parser.add_argument("--plots_dir", type=str, default="plots/representation_geometry")

    parser.add_argument("--model_names", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument(
        "--include_final",
        action="store_true",
        help="Also analyze the final model directory when explicit checkpoints are provided.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--layers", nargs="*", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--preference_only",
        action="store_true",
        help="Only compute preference-direction metrics/plots from cached or newly extracted representations.",
    )
    parser.add_argument(
        "--skip_preference",
        action="store_true",
        help="Skip preference-direction metrics/plots during the regular pipeline.",
    )
    parser.add_argument(
        "--preference_layers",
        nargs="*",
        type=int,
        default=None,
        help="Layer indices for preference-direction analysis. Defaults to --layers or all layers.",
    )
    parser.add_argument(
        "--pca2d_only",
        action="store_true",
        help="Only generate PCA-2D plots/variance CSV from cached or newly extracted representations.",
    )
    parser.add_argument(
        "--neighborhood_only",
        action="store_true",
        help="Only compute kNN neighborhood-agreement metrics/plots from cached or newly extracted representations.",
    )
    parser.add_argument(
        "--knn_neighbors",
        type=int,
        default=KNN_NEIGHBORS,
        help="Number of same-class nearest neighbors for neighborhood-agreement analysis.",
    )
    parser.add_argument(
        "--skip_pca2d",
        action="store_true",
        help="Skip PCA-2D projection plots when --plot is set.",
    )
    parser.add_argument(
        "--pca2d_layers",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Layer indices to show in PCA-2D plots. Defaults to representative "
            "input/early/middle/late/final layers, or --layers if provided."
        ),
    )
    parser.add_argument(
        "--pca2d_max_samples",
        type=int,
        default=PCA2D_MAX_SAMPLES,
        help="Maximum examples to plot/fit PCA per dataset/model/checkpoint.",
    )
    parser.add_argument(
        "--plot_from_results",
        action="store_true",
        help="Regenerate plots from existing CSVs in results_dir without recomputing metrics.",
    )
    parser.add_argument(
        "--skip_cka",
        action="store_true",
        help="Skip cross-model CKA when only layer trajectory plots/results are needed.",
    )
    return parser.parse_args()


def discover_model_names(models_dir: Path, requested: Optional[Sequence[str]]) -> List[str]:
    if requested is not None:
        return list(requested)
    if not models_dir.exists():
        warn(f"Models directory not found: {models_dir}")
        return []
    names = [path.name for path in models_dir.iterdir() if path.is_dir()]
    return sort_names_with_architecture_hint(names)


def discover_datasets(test_dir: Path, requested: Optional[Sequence[str]]) -> List[DatasetSpec]:
    if not test_dir.exists():
        warn(f"Test dataset directory not found: {test_dir}")
        return []

    specs: List[DatasetSpec] = []
    seen = set()
    if requested is None:
        paths = sorted(test_dir.glob("*.jsonl"))
    else:
        paths = []
        for raw_name in requested:
            name = raw_name[:-6] if raw_name.endswith(".jsonl") else raw_name
            direct_path = test_dir / f"{name}.jsonl"
            if direct_path.exists():
                paths.append(direct_path)
            elif name == "temporal_concord":
                fold_paths = sorted(test_dir.glob("temporal_concord_fold_*.jsonl"))
                if not fold_paths:
                    warn(f"No temporal_concord_fold_*.jsonl files found in {test_dir}")
                paths.extend(fold_paths)
            else:
                warn(f"Dataset not found in {test_dir}: {name}.jsonl")

    for path in paths:
        name = path.stem
        if name in seen:
            continue
        seen.add(name)
        specs.append(DatasetSpec(name=name, path=path, split="test"))
    return specs


def requested_checkpoint_labels(args: argparse.Namespace) -> List[str]:
    if args.checkpoints is None:
        return ["final"]
    labels = [normalize_checkpoint_label(value) for value in args.checkpoints]
    if args.include_final and "final" not in labels:
        labels.append("final")
    seen = set()
    unique = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            unique.append(label)
    return sorted(unique, key=checkpoint_sort_key)


def resolve_checkpoints(model_dir: Path, labels: Sequence[str]) -> List[CheckpointSpec]:
    if not model_dir.exists():
        warn(f"Model directory not found: {model_dir}")
        return []

    specs = []
    for label in labels:
        if label == "final":
            path = model_dir
        else:
            path = model_dir / label

        if not path.exists():
            warn(f"Skipping missing checkpoint: {path}")
            continue
        specs.append(CheckpointSpec(label=label, path=path, step=checkpoint_step(label)))
    return specs


def stratified_sample_df(df: pd.DataFrame, max_samples: Optional[int], seed: int) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    labels = df["acceptability"].astype(int).to_numpy()
    selected: List[int] = []

    for label in sorted(pd.unique(labels)):
        idx = np.flatnonzero(labels == label)
        if len(idx) == 0:
            continue
        target = max(1, int(round(max_samples * len(idx) / len(df))))
        target = min(target, len(idx))
        selected.extend(rng.choice(idx, size=target, replace=False).tolist())

    if len(selected) > max_samples:
        selected = rng.choice(np.asarray(selected), size=max_samples, replace=False).tolist()
    elif len(selected) < max_samples:
        remaining = np.setdiff1d(np.arange(len(df)), np.asarray(selected), assume_unique=False)
        if len(remaining) > 0:
            add = min(max_samples - len(selected), len(remaining))
            selected.extend(rng.choice(remaining, size=add, replace=False).tolist())

    selected = sorted(int(i) for i in selected)
    return df.iloc[selected].reset_index(drop=True)


def load_dataset(spec: DatasetSpec, max_samples: Optional[int], seed: int) -> pd.DataFrame:
    df = pd.read_json(spec.path, lines=True)
    required_cols = {"id", "sentence", "acceptability"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"{spec.path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["sentence"] = df["sentence"].astype(str)
    df["acceptability"] = df["acceptability"].astype(int)
    sample_seed = stable_seed(spec.name, spec.split, base_seed=seed)
    return stratified_sample_df(df, max_samples=max_samples, seed=sample_seed)


def sample_tag(max_samples: Optional[int], seed: int) -> str:
    if max_samples is None:
        return "all"
    return f"max{max_samples}_seed{seed}"


def representation_cache_path(
    representations_dir: Path,
    dataset_name: str,
    model_name: str,
    checkpoint_label: str,
    split: str,
    sample: str,
) -> Path:
    file_name = "__".join(
        [
            safe_name(split),
            safe_name(dataset_name),
            safe_name(model_name),
            safe_name(checkpoint_label),
            safe_name(sample),
        ]
    )
    return representations_dir / f"{file_name}.npz"


def load_cached_representations(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        ids = np.asarray(data["ids"], dtype=str)
        labels = np.asarray(data["labels"], dtype=np.int64)
        sentences = np.asarray(data["sentences"], dtype=str)
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    return ids, labels, sentences, embeddings


def save_cached_representations(
    path: Path,
    ids: Sequence[str],
    labels: Sequence[int],
    sentences: Sequence[str],
    embeddings: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.asarray(ids, dtype=np.str_),
        labels=np.asarray(labels, dtype=np.int64),
        sentences=np.asarray(sentences, dtype=np.str_),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )


def load_tokenizer(tokenizer_path: Path):
    if AutoTokenizer is None:
        raise ImportError("transformers is required for representation extraction.")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    if tokenizer.pad_token is None:
        fallback_token = tokenizer.eos_token or tokenizer.unk_token
        if fallback_token is None:
            raise ValueError("Tokenizer has no pad/eos/unk token available for padded batches.")
        tokenizer.pad_token = fallback_token
    return tokenizer


def load_model(checkpoint_path: Path, device: str):
    if AutoModelForCausalLM is None:
        raise ImportError("transformers is required for representation extraction.")
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint_path), output_hidden_states=True)
    model.eval()
    model.to(device)
    return model


def extract_and_cache_representations(
    model,
    tokenizer,
    device: str,
    batch_size: int,
    dataset: pd.DataFrame,
    cache_path: Path,
) -> None:
    sentences = dataset["sentence"].tolist()
    # utils.get_last_token_embeddings keeps hidden_states[0], the token embedding
    # output, as layer 0. Transformer block outputs therefore start at layer 1.
    embeddings = get_last_token_embeddings(
        model,
        sentences,
        tokenizer,
        device,
        batch_size=batch_size,
        include_embedding_matrix=True,
    )
    embeddings = embeddings.numpy().astype(np.float32, copy=False)
    save_cached_representations(
        cache_path,
        ids=dataset["id"].tolist(),
        labels=dataset["acceptability"].tolist(),
        sentences=sentences,
        embeddings=embeddings,
    )


def selected_layers(num_layers: int, requested_layers: Optional[Sequence[int]]) -> List[int]:
    if requested_layers is None:
        return list(range(num_layers))
    layers = []
    for layer in requested_layers:
        if 0 <= layer < num_layers:
            layers.append(int(layer))
        else:
            warn(f"Skipping unavailable layer {layer}; representation has {num_layers} layers.")
    return sorted(set(layers))


def selected_pca_layers(
    num_layers: int,
    requested_pca_layers: Optional[Sequence[int]],
    requested_metric_layers: Optional[Sequence[int]],
) -> List[int]:
    if requested_pca_layers is not None:
        return selected_layers(num_layers, requested_pca_layers)
    if requested_metric_layers is not None:
        return selected_layers(num_layers, requested_metric_layers)
    # A compact layer profile is more readable than one PCA panel per layer.
    # We keep layer 0 (embedding output) and the final layer, plus evenly spaced
    # intermediate layers. Duplicate rounded indices are removed.
    n_panels = min(5, num_layers)
    return sorted({int(round(value)) for value in np.linspace(0, num_layers - 1, n_panels)})


def stratified_sample_indices(labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels).astype(int)
    n = len(labels)
    if max_samples is None or max_samples <= 0 or n <= max_samples:
        return np.arange(n, dtype=int)

    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for label in sorted(pd.unique(labels)):
        idx = np.flatnonzero(labels == label)
        if len(idx) == 0:
            continue
        target = max(1, int(round(max_samples * len(idx) / n)))
        target = min(target, len(idx))
        selected.extend(rng.choice(idx, size=target, replace=False).tolist())

    if len(selected) > max_samples:
        selected = rng.choice(np.asarray(selected), size=max_samples, replace=False).tolist()
    elif len(selected) < max_samples:
        remaining = np.setdiff1d(np.arange(n), np.asarray(selected), assume_unique=False)
        if len(remaining) > 0:
            add = min(max_samples - len(selected), len(remaining))
            selected.extend(rng.choice(remaining, size=add, replace=False).tolist())

    return np.asarray(sorted(int(i) for i in selected), dtype=int)


def pca2d_projection(X: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """Project centered features onto PC1/PC2 and report explained variance.

    We center but do not z-score dimensions, matching the intrinsic-dimensionality
    preprocessing: coordinate variance is part of the representation geometry.
    """
    X = np.asarray(X, dtype=np.float64)
    if len(X) < 2 or X.ndim != 2:
        return np.zeros((len(X), 2), dtype=np.float64), math.nan, math.nan, math.nan

    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        U, singular_values, _ = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((len(X), 2), dtype=np.float64), math.nan, math.nan, math.nan

    coords = np.zeros((len(X), 2), dtype=np.float64)
    n_components = min(2, len(singular_values))
    if n_components:
        coords[:, :n_components] = U[:, :n_components] * singular_values[:n_components]

    eigvals = (singular_values**2) / max(len(X) - 1, 1)
    total = float(eigvals.sum())
    if total <= EPS:
        return coords, math.nan, math.nan, math.nan
    pc1 = float(eigvals[0] / total) if len(eigvals) > 0 else math.nan
    pc2 = float(eigvals[1] / total) if len(eigvals) > 1 else 0.0
    total_2d = float(np.nansum([pc1, pc2]))
    return coords, pc1, pc2, total_2d


def sample_rows(X: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if len(X) <= max_rows:
        return X
    idx = rng.choice(len(X), size=max_rows, replace=False)
    return X[np.sort(idx)]


def sample_rows_with_labels(
    X: np.ndarray,
    labels: np.ndarray,
    max_rows: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) <= max_rows:
        return X, labels
    idx = rng.choice(len(X), size=max_rows, replace=False)
    idx = np.sort(idx)
    return X[idx], labels[idx]


def vector_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= EPS:
        return math.nan
    return float(1.0 - np.dot(a, b) / denom)


def covariance_trace(X: np.ndarray) -> float:
    if len(X) < 2:
        return 0.0
    Xc = X - X.mean(axis=0, keepdims=True)
    return float(np.sum(Xc * Xc) / (len(X) - 1))


def mean_pairwise_cosine_distance(
    X: np.ndarray,
    rng: np.random.Generator,
    max_rows: int = PAIRWISE_WITHIN_MAX,
) -> float:
    if len(X) < 2:
        return math.nan
    Xs = sample_rows(X, max_rows=max_rows, rng=rng)
    if len(Xs) < 2:
        return math.nan
    distances = pairwise_distances(Xs, metric="cosine")
    tri = np.triu_indices(len(Xs), k=1)
    values = distances[tri]
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else math.nan


def mean_between_cosine_distance(
    X0: np.ndarray,
    X1: np.ndarray,
    rng: np.random.Generator,
) -> float:
    if len(X0) == 0 or len(X1) == 0:
        return math.nan
    X0s = sample_rows(X0, max_rows=PAIRWISE_BETWEEN_MAX_PER_CLASS, rng=rng)
    X1s = sample_rows(X1, max_rows=PAIRWISE_BETWEEN_MAX_PER_CLASS, rng=rng)
    distances = pairwise_distances(X0s, X1s, metric="cosine")
    values = distances[np.isfinite(distances)]
    return float(values.mean()) if len(values) else math.nan


def mmd_rbf_median_heuristic(
    X0: np.ndarray,
    X1: np.ndarray,
    rng: np.random.Generator,
) -> float:
    if len(X0) < 2 or len(X1) < 2:
        return math.nan
    X0s = sample_rows(X0, max_rows=MMD_MAX_PER_CLASS, rng=rng)
    X1s = sample_rows(X1, max_rows=MMD_MAX_PER_CLASS, rng=rng)
    X = np.vstack([X0s, X1s])
    sq = pairwise_distances(X, metric="sqeuclidean")
    tri = sq[np.triu_indices(len(X), k=1)]
    tri = tri[np.isfinite(tri) & (tri > EPS)]
    if len(tri) == 0:
        return math.nan
    sigma2 = float(np.median(tri))
    if sigma2 <= EPS:
        return math.nan

    n0 = len(X0s)
    K = np.exp(-sq / (2.0 * sigma2))
    K00 = K[:n0, :n0]
    K11 = K[n0:, n0:]
    K01 = K[:n0, n0:]

    def offdiag_mean(matrix: np.ndarray) -> float:
        n = len(matrix)
        if n < 2:
            return math.nan
        return float(matrix[np.triu_indices(n, k=1)].mean())

    k00 = offdiag_mean(K00)
    k11 = offdiag_mean(K11)
    k01 = float(K01.mean())
    if not np.isfinite(k00) or not np.isfinite(k11):
        return math.nan
    return float(k00 + k11 - 2.0 * k01)


def compute_class_geometry(X: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> Dict[str, float]:
    labels = labels.astype(int)
    X = np.asarray(X, dtype=np.float64)
    X0 = X[labels == 0]
    X1 = X[labels == 1]
    n0 = len(X0)
    n1 = len(X1)

    result = {
        "n_total": int(len(X)),
        "n_class_0": int(n0),
        "n_class_1": int(n1),
        "centroid_euclidean": math.nan,
        "centroid_euclidean_class_0": math.nan,
        "centroid_euclidean_class_1": math.nan,
        "centroid_cosine_distance": math.nan,
        "within_class_0_mean_cosine_distance": math.nan,
        "within_class_1_mean_cosine_distance": math.nan,
        "pooled_within_mean_cosine_distance": math.nan,
        "between_class_mean_cosine_distance": math.nan,
        "separation_ratio": math.nan,
        "fisher_ratio": math.nan,
        "fisher_ratio_class_0": math.nan,
        "fisher_ratio_class_1": math.nan,
        "silhouette_cosine": math.nan,
        "mmd_rbf": math.nan,
    }
    if n0 == 0 or n1 == 0:
        return result

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    mu_all = X.mean(axis=0)
    centroid_delta = mu1 - mu0
    centroid_class_0 = mu0 - mu_all
    centroid_class_1 = mu1 - mu_all
    result["centroid_euclidean"] = float(np.linalg.norm(centroid_delta))
    # The global centroid is the reference point for class-specific centroid
    # displacement. The original centroid_euclidean remains the between-class
    # distance ||mu_1 - mu_0||.
    result["centroid_euclidean_class_0"] = float(np.linalg.norm(centroid_class_0))
    result["centroid_euclidean_class_1"] = float(np.linalg.norm(centroid_class_1))
    result["centroid_cosine_distance"] = vector_cosine_distance(mu1, mu0)
    result["within_class_0_mean_cosine_distance"] = mean_pairwise_cosine_distance(X0, rng)
    result["within_class_1_mean_cosine_distance"] = mean_pairwise_cosine_distance(X1, rng)
    result["between_class_mean_cosine_distance"] = mean_between_cosine_distance(X0, X1, rng)

    pair_weights = np.asarray([n0 * max(n0 - 1, 0) / 2.0, n1 * max(n1 - 1, 0) / 2.0])
    within_values = np.asarray(
        [
            result["within_class_0_mean_cosine_distance"],
            result["within_class_1_mean_cosine_distance"],
        ],
        dtype=float,
    )
    valid = np.isfinite(within_values) & (pair_weights > 0)
    if valid.any():
        pooled = float(np.average(within_values[valid], weights=pair_weights[valid]))
        result["pooled_within_mean_cosine_distance"] = pooled
        if pooled > EPS and np.isfinite(result["between_class_mean_cosine_distance"]):
            result["separation_ratio"] = float(result["between_class_mean_cosine_distance"] / pooled)

    trace0 = covariance_trace(X0)
    trace1 = covariance_trace(X1)
    denom = trace0 + trace1 + EPS
    result["fisher_ratio"] = float(np.dot(centroid_delta, centroid_delta) / denom)
    # Class-specific Fisher contribution: class centroid displacement from the
    # pooled centroid normalized by that class' scatter.
    result["fisher_ratio_class_0"] = float(np.dot(centroid_class_0, centroid_class_0) / (trace0 + EPS))
    result["fisher_ratio_class_1"] = float(np.dot(centroid_class_1, centroid_class_1) / (trace1 + EPS))

    if len(X) >= 3 and n0 >= 2 and n1 >= 2:
        Xs, ys = sample_rows_with_labels(X, labels, max_rows=SILHOUETTE_MAX, rng=rng)
        if len(np.unique(ys)) == 2 and len(Xs) > len(np.unique(ys)):
            try:
                result["silhouette_cosine"] = float(silhouette_score(Xs, ys, metric="cosine"))
            except ValueError:
                result["silhouette_cosine"] = math.nan

    result["mmd_rbf"] = mmd_rbf_median_heuristic(X0, X1, rng)
    return result


def covariance_eigenvalues_centered(X: np.ndarray) -> np.ndarray:
    if len(X) < 2:
        return np.asarray([], dtype=np.float64)
    # We center but do not z-score dimensions: relative coordinate variance is
    # part of the representation geometry we want to describe, not remove.
    Xc = X.astype(np.float64, copy=False) - X.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    eigvals = (singular_values**2) / max(len(X) - 1, 1)
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvals[eigvals < 0] = 0.0
    return eigvals[np.isfinite(eigvals)]


def pca_rank_for_variance(eigvals: np.ndarray, threshold: float) -> float:
    total = float(eigvals.sum())
    if total <= EPS:
        return math.nan
    cumulative = np.cumsum(eigvals) / total
    return float(np.searchsorted(cumulative, threshold, side="left") + 1)


def effective_rank(eigvals: np.ndarray) -> float:
    total = float(eigvals.sum())
    if total <= EPS:
        return math.nan
    probs = eigvals / total
    probs = probs[probs > EPS]
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(np.exp(entropy))


def nearest_distances(X: np.ndarray, n_neighbors: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X)
    distances, _ = nn.kneighbors(X, return_distance=True)
    return distances


def jitter_unique(X: np.ndarray, scale: float = 1e-8) -> np.ndarray:
    """Break exact duplicate rows without changing macroscopic geometry."""
    if len(X) == 0:
        return X
    _, counts = np.unique(X, axis=0, return_counts=True)
    if np.all(counts == 1):
        return X
    rng = np.random.default_rng(0)
    return X + rng.normal(0.0, scale, size=X.shape)


_ESS_WARNED = False


def ess_id(X: np.ndarray) -> float:
    """Expected simplex skewness intrinsic dimension estimate from skdim."""
    global _ESS_WARNED
    if ESS is None:
        if not _ESS_WARNED:
            warn("scikit-dimension is not installed; ess_id will be NaN.")
            _ESS_WARNED = True
        return math.nan
    if len(X) < 3:
        return math.nan
    try:
        estimator = ESS()
        n_neighbors = min(ESS_N_NEIGHBORS, len(X) - 1)
        estimator.fit(jitter_unique(X), n_neighbors=n_neighbors)
        return float(getattr(estimator, "dimension_", math.nan))
    except Exception as exc:
        if not _ESS_WARNED:
            warn(f"ESS intrinsic-dimensionality estimation failed; ess_id will be NaN. First error: {exc}")
            _ESS_WARNED = True
        return math.nan


def twonn_id(X: np.ndarray) -> float:
    n = len(X)
    if n < 3:
        return math.nan
    distances = nearest_distances(X, n_neighbors=min(n, max(3, min(50, n))))
    logs = []
    for row in distances:
        positive = row[row > EPS]
        if len(positive) < 2:
            continue
        ratio = positive[1] / positive[0]
        if ratio > 1.0 + EPS:
            logs.append(math.log(ratio))
    if not logs:
        return math.nan
    mean_log = float(np.mean(logs))
    return float(1.0 / mean_log) if mean_log > EPS else math.nan


def mle_id(X: np.ndarray, k: int) -> float:
    n = len(X)
    if n <= k:
        return math.nan
    distances = nearest_distances(X, n_neighbors=min(n, max(k + 1, min(50, n))))
    estimates = []
    for row in distances:
        positive = row[row > EPS]
        if len(positive) < k:
            continue
        radii = positive[:k]
        outer = radii[-1]
        inner = radii[:-1]
        if outer <= EPS or np.any(inner <= EPS):
            continue
        denom = float(np.mean(np.log(outer / inner)))
        if denom > EPS:
            estimates.append(1.0 / denom)
    return float(np.mean(estimates)) if estimates else math.nan


def compute_intrinsic_dimensionality(
    X: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, float]:
    X = np.asarray(X, dtype=np.float64)
    Xs = sample_rows(X, max_rows=ID_MAX_SAMPLES, rng=rng)
    result = {
        "n_samples": int(len(X)),
        "n_samples_used": int(len(Xs)),
        "ambient_dim": int(X.shape[1]) if X.ndim == 2 else 0,
        "pca_rank_90": math.nan,
        "pca_rank_95": math.nan,
        "pca_rank_99": math.nan,
        "effective_rank": math.nan,
        "ess_id": math.nan,
        "twonn_id": math.nan,
        "mle_id_k5": math.nan,
        "mle_id_k10": math.nan,
        "mle_id_k20": math.nan,
    }
    if len(Xs) < 2 or Xs.ndim != 2:
        return result

    eigvals = covariance_eigenvalues_centered(Xs)
    result["pca_rank_90"] = pca_rank_for_variance(eigvals, 0.90)
    result["pca_rank_95"] = pca_rank_for_variance(eigvals, 0.95)
    result["pca_rank_99"] = pca_rank_for_variance(eigvals, 0.99)
    result["effective_rank"] = effective_rank(eigvals)

    # Centering leaves Euclidean nearest-neighbor distances unchanged, but using
    # the centered matrix here keeps PCA/ID preprocessing explicit and aligned.
    Xc = Xs - Xs.mean(axis=0, keepdims=True)
    X_ess = sample_rows(Xc, max_rows=ESS_MAX_SAMPLES, rng=rng)
    result["ess_id"] = ess_id(X_ess)
    result["twonn_id"] = twonn_id(Xc)
    for k in (5, 10, 20):
        result[f"mle_id_k{k}"] = mle_id(Xc, k=k)
    return result


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    if len(X) < 2 or len(Y) < 2 or len(X) != len(Y):
        return math.nan
    Xc = X.astype(np.float64, copy=False) - X.mean(axis=0, keepdims=True)
    Yc = Y.astype(np.float64, copy=False) - Y.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    denominator = np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro")
    if denominator <= EPS:
        return math.nan
    return float(numerator / denominator)


def align_ids(ids_a: np.ndarray, ids_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    index_b = {id_value: idx for idx, id_value in enumerate(ids_b)}
    idx_a = []
    idx_b = []
    for i, id_value in enumerate(ids_a):
        j = index_b.get(id_value)
        if j is not None:
            idx_a.append(i)
            idx_b.append(j)
    return np.asarray(idx_a, dtype=int), np.asarray(idx_b, dtype=int)


def common_id_order(model_ids: Dict[str, np.ndarray]) -> List[str]:
    if not model_ids:
        return []
    models = list(model_ids)
    common = set(model_ids[models[0]].tolist())
    for model in models[1:]:
        common.intersection_update(model_ids[model].tolist())
    return [id_value for id_value in model_ids[models[0]].tolist() if id_value in common]


def closest_layer_by_depth(
    target_depth: float,
    num_layers: int,
    requested_layers_arg: Optional[Sequence[int]],
) -> int:
    layers = selected_layers(num_layers, requested_layers_arg)
    return min(layers, key=lambda layer: abs(layer_depth(layer, num_layers) - target_depth))


def knn_neighbor_sets(
    X: np.ndarray,
    k: int = KNN_NEIGHBORS,
    metric: str = KNN_METRIC,
) -> Tuple[List[set], int]:
    if len(X) < 2:
        return [], 0
    k_effective = min(k, len(X) - 1)
    n_neighbors = min(len(X), k_effective + 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(X)
    indices = nn.kneighbors(X, return_distance=False)
    neighbor_sets = []
    for row_idx, row in enumerate(indices):
        neighbors = [int(idx) for idx in row if int(idx) != row_idx]
        neighbor_sets.append(set(neighbors[:k_effective]))
    return neighbor_sets, k_effective


def neighbor_overlap_stats(
    neighbors_a: Sequence[set],
    neighbors_b: Sequence[set],
    k_effective: int,
) -> Dict[str, float]:
    if not neighbors_a or not neighbors_b or len(neighbors_a) != len(neighbors_b) or k_effective <= 0:
        return {
            "mean_jaccard": math.nan,
            "std_jaccard": math.nan,
            "mean_overlap_count": math.nan,
            "mean_overlap_fraction": math.nan,
            "random_overlap_fraction": math.nan,
            "normalized_overlap_fraction": math.nan,
        }

    jaccards = []
    overlaps = []
    for set_a, set_b in zip(neighbors_a, neighbors_b):
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        jaccards.append(intersection / union if union else math.nan)
        overlaps.append(intersection)

    jaccards = np.asarray(jaccards, dtype=np.float64)
    overlaps = np.asarray(overlaps, dtype=np.float64)
    jaccards = jaccards[np.isfinite(jaccards)]
    overlaps = overlaps[np.isfinite(overlaps)]
    n_pool = max(len(neighbors_a) - 1, 1)
    random_overlap = min(1.0, k_effective / n_pool)
    mean_overlap_fraction = float(overlaps.mean() / k_effective) if len(overlaps) else math.nan
    if np.isfinite(mean_overlap_fraction) and random_overlap < 1.0:
        normalized = (mean_overlap_fraction - random_overlap) / (1.0 - random_overlap)
    else:
        normalized = math.nan

    return {
        "mean_jaccard": float(jaccards.mean()) if len(jaccards) else math.nan,
        "std_jaccard": float(jaccards.std(ddof=1)) if len(jaccards) > 1 else math.nan,
        "mean_overlap_count": float(overlaps.mean()) if len(overlaps) else math.nan,
        "mean_overlap_fraction": mean_overlap_fraction,
        "random_overlap_fraction": float(random_overlap),
        "normalized_overlap_fraction": float(normalized) if np.isfinite(normalized) else math.nan,
    }


def temporal_fold_number(dataset: str) -> float:
    match = re.search(r"temporal_concord_fold_(\d+)", dataset)
    return float(match.group(1)) if match else math.nan


def add_temporal_fold_aggregates(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    count_cols: Sequence[str],
) -> pd.DataFrame:
    if df.empty or "dataset" not in df.columns:
        return df

    output = df.copy()
    if "is_aggregate" not in output.columns:
        output["is_aggregate"] = False
    if "n_folds" not in output.columns:
        output["n_folds"] = math.nan

    folds = output[output["dataset"].astype(str).str.startswith("temporal_concord_fold_")].copy()
    if folds.empty:
        return output

    group_cols = [col for col in group_cols if col in folds.columns]
    count_cols = [col for col in count_cols if col in folds.columns]
    excluded = set(group_cols) | set(count_cols) | {"fold", "is_aggregate", "n_folds"}
    numeric_cols = list(folds.select_dtypes(include=[np.number]).columns)
    metric_cols = [col for col in numeric_cols if col not in excluded]

    agg_spec = {col: "sum" for col in count_cols}
    agg_spec.update({col: "mean" for col in metric_cols})
    grouped = folds.groupby(group_cols, dropna=False)
    aggregate = grouped.agg(agg_spec).reset_index()
    sizes = grouped.size().reset_index(name="n_folds")
    aggregate = aggregate.merge(sizes, on=group_cols, how="left")
    aggregate["dataset"] = "temporal_concord"
    aggregate["fold"] = math.nan
    aggregate["is_aggregate"] = True
    return pd.concat([output, aggregate], ignore_index=True, sort=False)


def compute_layer_metrics(
    representation_index: Dict[Tuple[str, str, str], Path],
    dataset_specs: Sequence[DatasetSpec],
    requested_layers_arg: Optional[Sequence[int]],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    class_records = []
    id_records = []

    for dataset_name, model_name, checkpoint_label in tqdm(
        sorted(representation_index), desc="Geometry metrics"
    ):
        cache_path = representation_index[(dataset_name, model_name, checkpoint_label)]
        ids, labels, _, embeddings = load_cached_representations(cache_path)
        num_layers = embeddings.shape[1]
        layers = selected_layers(num_layers, requested_layers_arg)
        split = next((spec.split for spec in dataset_specs if spec.name == dataset_name), "test")

        for layer in layers:
            rng = np.random.default_rng(
                stable_seed(dataset_name, model_name, checkpoint_label, layer, "geometry", base_seed=seed)
            )
            X = embeddings[:, layer, :]
            base = {
                "split": split,
                "dataset": dataset_name,
                "fold": temporal_fold_number(dataset_name),
                "model": model_name,
                "checkpoint": checkpoint_label,
                "checkpoint_step": checkpoint_step(checkpoint_label),
                "layer": layer,
                "layer_depth": layer_depth(layer, num_layers),
            }

            class_records.append({**base, **compute_class_geometry(X, labels, rng)})

            subsets = {
                "all": np.ones(len(labels), dtype=bool),
                "class_0": labels == 0,
                "class_1": labels == 1,
            }
            for subset_name, mask in subsets.items():
                subset_rng = np.random.default_rng(
                    stable_seed(
                        dataset_name,
                        model_name,
                        checkpoint_label,
                        layer,
                        subset_name,
                        "intrinsic",
                        base_seed=seed,
                    )
                )
                id_records.append(
                    {
                        **base,
                        "subset": subset_name,
                        **compute_intrinsic_dimensionality(X[mask], subset_rng),
                    }
                )

    class_df = pd.DataFrame(class_records)
    id_df = pd.DataFrame(id_records)

    class_df = add_temporal_fold_aggregates(
        class_df,
        group_cols=["split", "model", "checkpoint", "checkpoint_step", "layer", "layer_depth"],
        count_cols=["n_total", "n_class_0", "n_class_1"],
    )
    id_df = add_temporal_fold_aggregates(
        id_df,
        group_cols=[
            "split",
            "model",
            "checkpoint",
            "checkpoint_step",
            "layer",
            "layer_depth",
            "subset",
        ],
        count_cols=["n_samples", "n_samples_used"],
    )
    return class_df, id_df


def compute_cross_model_cka(
    representation_index: Dict[Tuple[str, str, str], Path],
    requested_layers_arg: Optional[Sequence[int]],
) -> pd.DataFrame:
    records = []
    keys = sorted(representation_index)
    dataset_names = sorted({key[0] for key in keys})
    checkpoint_labels = sorted({key[2] for key in keys}, key=checkpoint_sort_key)

    for dataset_name in tqdm(dataset_names, desc="Cross-model CKA"):
        for checkpoint_label in checkpoint_labels:
            models_for_dataset_checkpoint = sorted(
                {
                    model
                    for ds, model, ckpt in keys
                    if ds == dataset_name and ckpt == checkpoint_label
                }
            )
            model_entries = [
                (model, representation_index[(dataset_name, model, checkpoint_label)])
                for model in models_for_dataset_checkpoint
            ]
            for (model_a, path_a), (model_b, path_b) in combinations(model_entries, 2):
                ids_a, labels_a, _, embeddings_a = load_cached_representations(path_a)
                ids_b, labels_b, _, embeddings_b = load_cached_representations(path_b)
                idx_a, idx_b = align_ids(ids_a, ids_b)
                if len(idx_a) < 2:
                    warn(
                        f"Skipping CKA for {dataset_name}/{checkpoint_label}/{model_a}/{model_b}: "
                        "fewer than 2 aligned ids."
                    )
                    continue

                aligned_labels_a = labels_a[idx_a]
                aligned_labels_b = labels_b[idx_b]
                if not np.array_equal(aligned_labels_a, aligned_labels_b):
                    warn(
                        f"Labels differ after id alignment for {dataset_name}/{model_a}/{model_b}; "
                        "CKA is still computed on aligned sentences."
                    )

                layers_a = selected_layers(embeddings_a.shape[1], requested_layers_arg)
                layers_b = selected_layers(embeddings_b.shape[1], requested_layers_arg)
                for layer_a in layers_a:
                    Xa = embeddings_a[idx_a, layer_a, :]
                    for layer_b in layers_b:
                        Yb = embeddings_b[idx_b, layer_b, :]
                        records.append(
                            {
                                "split": "test",
                                "dataset": dataset_name,
                                "fold": temporal_fold_number(dataset_name),
                                "checkpoint_a": checkpoint_label,
                                "checkpoint_b": checkpoint_label,
                                "checkpoint_step_a": checkpoint_step(checkpoint_label),
                                "checkpoint_step_b": checkpoint_step(checkpoint_label),
                                "model_a": model_a,
                                "model_b": model_b,
                                "layer_a": layer_a,
                                "layer_b": layer_b,
                                "layer_depth_a": layer_depth(layer_a, embeddings_a.shape[1]),
                                "layer_depth_b": layer_depth(layer_b, embeddings_b.shape[1]),
                                "n_aligned": int(len(idx_a)),
                                "cka": linear_cka(Xa, Yb),
                            }
                        )

    cka_df = pd.DataFrame(records)
    return add_temporal_fold_aggregates(
        cka_df,
        group_cols=[
            "split",
            "checkpoint_a",
            "checkpoint_b",
            "checkpoint_step_a",
            "checkpoint_step_b",
            "model_a",
            "model_b",
            "layer_a",
            "layer_b",
            "layer_depth_a",
            "layer_depth_b",
        ],
        count_cols=["n_aligned"],
    )


def compute_knn_neighborhood_agreement(
    representation_index: Dict[Tuple[str, str, str], Path],
    requested_layers_arg: Optional[Sequence[int]],
    knn_neighbors: int = KNN_NEIGHBORS,
) -> pd.DataFrame:
    records = []
    keys = sorted(representation_index)
    dataset_names = sorted({key[0] for key in keys})
    checkpoint_labels = sorted({key[2] for key in keys}, key=checkpoint_sort_key)

    for dataset_name in tqdm(dataset_names, desc="kNN agreement"):
        for checkpoint_label in checkpoint_labels:
            model_names = sort_names_with_architecture_hint(
                {
                    model
                    for ds, model, ckpt in keys
                    if ds == dataset_name and ckpt == checkpoint_label
                }
            )
            if len(model_names) < 2:
                continue

            loaded = {}
            for model_name in model_names:
                ids, labels, _, embeddings = load_cached_representations(
                    representation_index[(dataset_name, model_name, checkpoint_label)]
                )
                loaded[model_name] = {
                    "ids": ids,
                    "labels": labels,
                    "embeddings": embeddings,
                    "num_layers": embeddings.shape[1],
                }

            shared_ids = common_id_order({model: loaded[model]["ids"] for model in model_names})
            if len(shared_ids) < 2:
                warn(f"Skipping kNN agreement for {dataset_name}/{checkpoint_label}: no shared ids.")
                continue

            aligned = {}
            for model_name in model_names:
                id_index = {id_value: idx for idx, id_value in enumerate(loaded[model_name]["ids"])}
                idx = np.asarray([id_index[id_value] for id_value in shared_ids], dtype=int)
                aligned[model_name] = {
                    "labels": loaded[model_name]["labels"][idx],
                    "embeddings": loaded[model_name]["embeddings"][idx],
                    "num_layers": loaded[model_name]["num_layers"],
                }

            reference_labels = aligned[model_names[0]]["labels"]
            for model_name in model_names[1:]:
                if not np.array_equal(reference_labels, aligned[model_name]["labels"]):
                    warn(
                        f"Labels differ after id alignment for {dataset_name}/{checkpoint_label}/{model_name}; "
                        "kNN agreement uses labels from the first model."
                    )

            neighbor_cache = {}
            for model_name in model_names:
                model_layers = selected_layers(aligned[model_name]["num_layers"], requested_layers_arg)
                for layer in model_layers:
                    for class_subset, label in [("class_0", 0), ("class_1", 1)]:
                        mask = reference_labels == label
                        X = aligned[model_name]["embeddings"][mask, layer, :]
                        neighbors, k_effective = knn_neighbor_sets(X, k=knn_neighbors, metric=KNN_METRIC)
                        neighbor_cache[(model_name, layer, class_subset)] = {
                            "neighbors": neighbors,
                            "k_effective": k_effective,
                            "n_items": int(mask.sum()),
                        }

            for model_a, model_b in combinations(model_names, 2):
                layers_a = selected_layers(aligned[model_a]["num_layers"], requested_layers_arg)
                for layer_a in layers_a:
                    depth_a = layer_depth(layer_a, aligned[model_a]["num_layers"])
                    layer_b = closest_layer_by_depth(
                        depth_a,
                        aligned[model_b]["num_layers"],
                        requested_layers_arg,
                    )
                    depth_b = layer_depth(layer_b, aligned[model_b]["num_layers"])
                    for class_subset in ["class_0", "class_1"]:
                        cached_a = neighbor_cache[(model_a, layer_a, class_subset)]
                        cached_b = neighbor_cache[(model_b, layer_b, class_subset)]
                        k_effective = min(cached_a["k_effective"], cached_b["k_effective"])
                        stats = neighbor_overlap_stats(
                            cached_a["neighbors"],
                            cached_b["neighbors"],
                            k_effective,
                        )
                        records.append(
                            {
                                "split": "test",
                                "dataset": dataset_name,
                                "fold": temporal_fold_number(dataset_name),
                                "checkpoint": checkpoint_label,
                                "checkpoint_step": checkpoint_step(checkpoint_label),
                                "model_a": model_a,
                                "model_b": model_b,
                                "layer_a": layer_a,
                                "layer_b": layer_b,
                                "layer_depth": float((depth_a + depth_b) / 2.0),
                                "layer_depth_a": depth_a,
                                "layer_depth_b": depth_b,
                                "class_subset": class_subset,
                                "n_aligned": int(len(shared_ids)),
                                "n_items": int(min(cached_a["n_items"], cached_b["n_items"])),
                                "k": int(knn_neighbors),
                                "k_effective": int(k_effective),
                                "distance_metric": KNN_METRIC,
                                **stats,
                            }
                        )

    knn_df = pd.DataFrame(records)
    return add_temporal_fold_aggregates(
        knn_df,
        group_cols=[
            "split",
            "checkpoint",
            "checkpoint_step",
            "model_a",
            "model_b",
            "layer_a",
            "layer_b",
            "layer_depth",
            "layer_depth_a",
            "layer_depth_b",
            "class_subset",
            "k",
            "k_effective",
            "distance_metric",
        ],
        count_cols=["n_aligned", "n_items"],
    )


def compute_checkpoint_drift(
    representation_index: Dict[Tuple[str, str, str], Path],
    requested_layers_arg: Optional[Sequence[int]],
) -> pd.DataFrame:
    records = []
    dataset_names = sorted({key[0] for key in representation_index})
    model_names = sorted({key[1] for key in representation_index})

    for dataset_name in tqdm(dataset_names, desc="Checkpoint drift"):
        for model_name in model_names:
            checkpoint_labels = sorted(
                [
                    ckpt
                    for ds, model, ckpt in representation_index
                    if ds == dataset_name and model == model_name
                ],
                key=checkpoint_sort_key,
            )
            if len(checkpoint_labels) < 2:
                continue

            loaded = {
                ckpt: load_cached_representations(representation_index[(dataset_name, model_name, ckpt)])
                for ckpt in checkpoint_labels
            }

            for checkpoint_a, checkpoint_b in combinations(checkpoint_labels, 2):
                ids_a, labels_a, _, embeddings_a = loaded[checkpoint_a]
                ids_b, labels_b, _, embeddings_b = loaded[checkpoint_b]
                idx_a, idx_b = align_ids(ids_a, ids_b)
                if len(idx_a) < 2:
                    continue
                max_layers = min(embeddings_a.shape[1], embeddings_b.shape[1])
                layers = [
                    layer
                    for layer in selected_layers(max_layers, requested_layers_arg)
                    if layer < embeddings_a.shape[1] and layer < embeddings_b.shape[1]
                ]
                for layer in layers:
                    records.append(
                        {
                            "split": "test",
                            "dataset": dataset_name,
                            "fold": temporal_fold_number(dataset_name),
                            "model": model_name,
                            "metric": "cka_same_layer",
                            "checkpoint_a": checkpoint_a,
                            "checkpoint_b": checkpoint_b,
                            "checkpoint_step_a": checkpoint_step(checkpoint_a),
                            "checkpoint_step_b": checkpoint_step(checkpoint_b),
                            "reference_checkpoint": math.nan,
                            "layer": layer,
                            "layer_depth": layer_depth(layer, max_layers),
                            "label": math.nan,
                            "n_aligned": int(len(idx_a)),
                            "cka": linear_cka(
                                embeddings_a[idx_a, layer, :],
                                embeddings_b[idx_b, layer, :],
                            ),
                            "centroid_drift_euclidean": math.nan,
                        }
                    )

            reference = "final" if "final" in checkpoint_labels else checkpoint_labels[-1]
            if reference != "final":
                warn(
                    f"No final checkpoint selected for {model_name}/{dataset_name}; "
                    f"centroid drift uses {reference} as the reference."
                )

            ids_ref, labels_ref, _, embeddings_ref = loaded[reference]
            for checkpoint_label in checkpoint_labels:
                ids_cur, labels_cur, _, embeddings_cur = loaded[checkpoint_label]
                idx_cur, idx_ref = align_ids(ids_cur, ids_ref)
                if len(idx_cur) < 2:
                    continue
                max_layers = min(embeddings_cur.shape[1], embeddings_ref.shape[1])
                layers = [
                    layer
                    for layer in selected_layers(max_layers, requested_layers_arg)
                    if layer < embeddings_cur.shape[1] and layer < embeddings_ref.shape[1]
                ]
                for layer in layers:
                    for label in (0, 1):
                        mask_cur = labels_cur[idx_cur] == label
                        mask_ref = labels_ref[idx_ref] == label
                        if mask_cur.sum() == 0 or mask_ref.sum() == 0:
                            drift = math.nan
                        else:
                            mu_cur = embeddings_cur[idx_cur[mask_cur], layer, :].mean(axis=0)
                            mu_ref = embeddings_ref[idx_ref[mask_ref], layer, :].mean(axis=0)
                            drift = float(np.linalg.norm(mu_cur - mu_ref))
                        records.append(
                            {
                                "split": "test",
                                "dataset": dataset_name,
                                "fold": temporal_fold_number(dataset_name),
                                "model": model_name,
                                "metric": "centroid_drift_to_reference",
                                "checkpoint_a": checkpoint_label,
                                "checkpoint_b": reference,
                                "checkpoint_step_a": checkpoint_step(checkpoint_label),
                                "checkpoint_step_b": checkpoint_step(reference),
                                "reference_checkpoint": reference,
                                "layer": layer,
                                "layer_depth": layer_depth(layer, max_layers),
                                "label": int(label),
                                "n_aligned": int(len(idx_cur)),
                                "cka": math.nan,
                                "centroid_drift_euclidean": drift,
                            }
                        )

    drift_df = pd.DataFrame(records)
    return add_temporal_fold_aggregates(
        drift_df,
        group_cols=[
            "split",
            "model",
            "metric",
            "checkpoint_a",
            "checkpoint_b",
            "checkpoint_step_a",
            "checkpoint_step_b",
            "reference_checkpoint",
            "layer",
            "layer_depth",
            "label",
        ],
        count_cols=["n_aligned"],
    )


def ensure_representation_caches(
    args: argparse.Namespace,
    model_names: Sequence[str],
    checkpoint_labels: Sequence[str],
    dataset_specs: Sequence[DatasetSpec],
    datasets: Dict[str, pd.DataFrame],
) -> Dict[Tuple[str, str, str], Path]:
    representations_dir = Path(args.representations_dir)
    representations_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)
    sample = sample_tag(args.max_samples, args.seed)
    representation_index: Dict[Tuple[str, str, str], Path] = {}
    tokenizer = None

    for model_name in model_names:
        model_dir = models_dir / model_name
        checkpoint_specs = resolve_checkpoints(model_dir, checkpoint_labels)
        if not checkpoint_specs:
            continue

        for checkpoint in checkpoint_specs:
            missing = []
            for spec in dataset_specs:
                cache_path = representation_cache_path(
                    representations_dir,
                    dataset_name=spec.name,
                    model_name=model_name,
                    checkpoint_label=checkpoint.label,
                    split=spec.split,
                    sample=sample,
                )
                key = (spec.name, model_name, checkpoint.label)
                if cache_path.exists() and not args.force_recompute:
                    representation_index[key] = cache_path
                else:
                    missing.append((spec, cache_path))

            if not missing:
                continue

            try:
                if tokenizer is None:
                    tokenizer = load_tokenizer(Path(args.tokenizer_path))
                model = load_model(checkpoint.path, args.device)
            except Exception as exc:
                warn(f"Could not load {model_name}/{checkpoint.label}: {exc}")
                continue

            for spec, cache_path in missing:
                try:
                    extract_and_cache_representations(
                        model=model,
                        tokenizer=tokenizer,
                        device=args.device,
                        batch_size=args.batch_size,
                        dataset=datasets[spec.name],
                        cache_path=cache_path,
                    )
                    representation_index[(spec.name, model_name, checkpoint.label)] = cache_path
                except Exception as exc:
                    warn(f"Could not extract {model_name}/{checkpoint.label}/{spec.name}: {exc}")

            del model
            if torch is not None and args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    return representation_index


def with_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if df.empty and len(df.columns) == 0:
        return pd.DataFrame(columns=columns)
    output = df.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = math.nan
    extra_columns = [column for column in output.columns if column not in columns]
    return output[list(columns) + extra_columns]


def write_csv(df: pd.DataFrame, path: Path, columns: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        df = with_columns(df, columns)
    df.to_csv(path, index=False)


def pca_plot_dataset_name(dataset_name: str) -> str:
    if str(dataset_name).startswith("temporal_concord_fold_"):
        return "temporal_concord"
    return str(dataset_name)


def grouped_representation_paths(
    representation_index: Dict[Tuple[str, str, str], Path]
) -> Dict[Tuple[str, str, str], List[Tuple[str, Path]]]:
    grouped: Dict[Tuple[str, str, str], List[Tuple[str, Path]]] = {}
    for dataset_name, model_name, checkpoint_label in sorted(representation_index):
        plot_dataset = pca_plot_dataset_name(dataset_name)
        key = (plot_dataset, model_name, checkpoint_label)
        grouped.setdefault(key, []).append((dataset_name, representation_index[(dataset_name, model_name, checkpoint_label)]))
    return grouped


def load_grouped_representations(paths: Sequence[Tuple[str, Path]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids_list = []
    labels_list = []
    embeddings_list = []
    dataset_names = []
    expected_shape = None

    for dataset_name, path in paths:
        ids, labels, _, embeddings = load_cached_representations(path)
        shape = embeddings.shape[1:]
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            warn(f"Skipping {path}; expected representation shape {expected_shape}, found {shape}.")
            continue
        ids_list.append(np.asarray([f"{dataset_name}::{id_value}" for id_value in ids], dtype=str))
        labels_list.append(labels)
        embeddings_list.append(embeddings)
        dataset_names.append(dataset_name)

    if not embeddings_list:
        return (
            np.asarray([], dtype=str),
            np.asarray([], dtype=int),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=str),
        )
    return (
        np.concatenate(ids_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(embeddings_list, axis=0),
        np.asarray(dataset_names, dtype=str),
    )


def plot_pca2d_for_representations(
    representation_index: Dict[Tuple[str, str, str], Path],
    dataset_specs: Sequence[DatasetSpec],
    requested_layers_arg: Optional[Sequence[int]],
    requested_pca_layers_arg: Optional[Sequence[int]],
    max_samples: int,
    seed: int,
    plots_dir: Path,
) -> pd.DataFrame:
    if not representation_index:
        return pd.DataFrame(columns=PCA2D_COLUMNS)

    pca_dir = plots_dir / "pca2d"
    pca_dir.mkdir(parents=True, exist_ok=True)
    split_by_dataset = {spec.name: spec.split for spec in dataset_specs}
    colors = {0: sns.color_palette("tab10", n_colors=3)[1], 1: sns.color_palette("tab10", n_colors=3)[2]}
    records = []

    for (dataset, model, checkpoint), paths in tqdm(
        sorted(grouped_representation_paths(representation_index).items()),
        desc="PCA-2D plots",
    ):
        _, labels, embeddings, source_datasets = load_grouped_representations(paths)
        if len(labels) == 0 or embeddings.size == 0:
            continue
        num_layers = embeddings.shape[1]
        layers = selected_pca_layers(num_layers, requested_pca_layers_arg, requested_layers_arg)
        if not layers:
            continue

        sample_seed = stable_seed(dataset, model, checkpoint, "pca2d", base_seed=seed)
        sample_idx = stratified_sample_indices(labels, max_samples=max_samples, seed=sample_seed)
        labels_sample = labels[sample_idx].astype(int)
        source_list = sorted(str(value) for value in source_datasets.tolist())
        split_values = sorted({split_by_dataset.get(name, "test") for name in source_list})
        split = split_values[0] if len(split_values) == 1 else ";".join(split_values)

        ncols = min(3, len(layers))
        nrows = int(math.ceil(len(layers) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.4 * ncols, 4.2 * nrows))
        axes = np.atleast_1d(axes).ravel()

        for ax, layer in zip(axes, layers):
            X = embeddings[sample_idx, layer, :]
            coords, pc1, pc2, pca2 = pca2d_projection(X)
            for label in [0, 1]:
                mask = labels_sample == label
                if not mask.any():
                    continue
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=16,
                    alpha=0.65,
                    color=colors[label],
                    edgecolors="none",
                    label=f"class_{label}",
                )
            style_plain_ax(ax, "PC1", "PC2")
            ax.set_title(
                f"Layer {layer} ({layer_depth(layer, num_layers):.2f})\n"
                f"PC1+PC2 = {pca2 * 100:.1f}%",
                fontsize=13,
            )
            records.append(
                {
                    "split": split,
                    "dataset": dataset,
                    "source_datasets": ";".join(source_list),
                    "model": model,
                    "checkpoint": checkpoint,
                    "checkpoint_step": checkpoint_step(checkpoint),
                    "layer": layer,
                    "layer_depth": layer_depth(layer, num_layers),
                    "n_total": int(len(labels)),
                    "n_plotted": int(len(sample_idx)),
                    "pc1_variance_ratio": pc1,
                    "pc2_variance_ratio": pc2,
                    "pca2d_variance_ratio": pca2,
                    "is_aggregate": dataset == "temporal_concord" and len(source_list) > 1,
                    "n_folds": float(len(source_list)) if dataset == "temporal_concord" and len(source_list) > 1 else math.nan,
                }
            )

        for ax in axes[len(layers) :]:
            ax.remove()

        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[label],
                       markersize=6, label=f"class_{label}")
            for label in [0, 1]
        ]
        fig.suptitle(
            f"PCA-2D representation projection — {dataset} — {short_model_name(model)} ({checkpoint})",
            fontsize=16,
        )
        fig.legend(
            handles,
            [handle.get_label() for handle in handles],
            loc="lower center",
            ncol=2,
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.tight_layout(rect=[0, 0.08, 1, 0.94])
        out = pca_dir / (
            f"geometry_pca2d_{safe_name(dataset)}_{safe_name(short_model_name(model))}_"
            f"{checkpoint_plot_label(checkpoint)}.svg"
        )
        fig.savefig(out, dpi=220)
        plt.close(fig)

    return pd.DataFrame(records)


def vector_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= EPS:
        return math.nan
    return float(np.dot(a, b) / denom)


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return math.nan
    ac = a[mask] - a[mask].mean()
    bc = b[mask] - b[mask].mean()
    return vector_cosine_similarity(ac, bc)


def preference_direction_stats(
    X: np.ndarray,
    labels: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    labels = labels.astype(int)
    X = np.asarray(X, dtype=np.float64)
    X0 = X[labels == 0]
    X1 = X[labels == 1]
    n0 = len(X0)
    n1 = len(X1)
    stats = {
        "n_total": int(len(X)),
        "n_class_0": int(n0),
        "n_class_1": int(n1),
        "direction_norm": math.nan,
        "score_mean_class_0": math.nan,
        "score_mean_class_1": math.nan,
        "score_gap": math.nan,
        "score_std_class_0": math.nan,
        "score_std_class_1": math.nan,
        "score_pooled_std": math.nan,
        "score_cohens_d": math.nan,
    }
    if n0 == 0 or n1 == 0:
        return stats, np.asarray([], dtype=np.float64), np.full(len(X), math.nan)

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    direction = mu1 - mu0
    direction_norm = float(np.linalg.norm(direction))
    stats["direction_norm"] = direction_norm
    if direction_norm <= EPS:
        return stats, np.asarray([], dtype=np.float64), np.full(len(X), math.nan)

    unit_direction = direction / direction_norm
    # Scores are descriptive projections onto the class-centroid direction.
    # We center by the pooled centroid so zero is the dataset/model/layer center.
    scores = (X - X.mean(axis=0, keepdims=True)) @ unit_direction
    scores0 = scores[labels == 0]
    scores1 = scores[labels == 1]
    stats["score_mean_class_0"] = float(scores0.mean())
    stats["score_mean_class_1"] = float(scores1.mean())
    stats["score_gap"] = float(scores1.mean() - scores0.mean())
    stats["score_std_class_0"] = float(scores0.std(ddof=1)) if n0 > 1 else math.nan
    stats["score_std_class_1"] = float(scores1.std(ddof=1)) if n1 > 1 else math.nan

    if n0 > 1 and n1 > 1:
        pooled_var = (
            (n0 - 1) * (stats["score_std_class_0"] ** 2)
            + (n1 - 1) * (stats["score_std_class_1"] ** 2)
        ) / max(n0 + n1 - 2, 1)
        pooled_std = math.sqrt(max(pooled_var, 0.0))
        stats["score_pooled_std"] = pooled_std
        if pooled_std > EPS:
            stats["score_cohens_d"] = float(stats["score_gap"] / pooled_std)

    return stats, unit_direction.astype(np.float32), scores.astype(np.float32)


def compute_preference_direction_analysis(
    representation_index: Dict[Tuple[str, str, str], Path],
    dataset_specs: Sequence[DatasetSpec],
    requested_layers_arg: Optional[Sequence[int]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_by_dataset = {spec.name: spec.split for spec in dataset_specs}
    direction_records = []
    score_index = {}
    direction_index = {}

    for (dataset, model, checkpoint), paths in tqdm(
        sorted(grouped_representation_paths(representation_index).items()),
        desc="Preference directions",
    ):
        ids, labels, embeddings, source_datasets = load_grouped_representations(paths)
        if len(labels) == 0 or embeddings.size == 0:
            continue
        num_layers = embeddings.shape[1]
        layers = selected_layers(num_layers, requested_layers_arg)
        source_list = sorted(str(value) for value in source_datasets.tolist())
        split_values = sorted({split_by_dataset.get(name, "test") for name in source_list})
        split = split_values[0] if len(split_values) == 1 else ";".join(split_values)
        is_aggregate = dataset == "temporal_concord" and len(source_list) > 1
        n_folds = float(len(source_list)) if is_aggregate else math.nan

        for layer in layers:
            X = embeddings[:, layer, :]
            stats, unit_direction, scores = preference_direction_stats(X, labels)
            base = {
                "split": split,
                "dataset": dataset,
                "source_datasets": ";".join(source_list),
                "model": model,
                "checkpoint": checkpoint,
                "checkpoint_step": checkpoint_step(checkpoint),
                "layer": layer,
                "layer_depth": layer_depth(layer, num_layers),
                "is_aggregate": is_aggregate,
                "n_folds": n_folds,
            }
            direction_records.append({**base, **stats})
            if len(unit_direction) == 0:
                continue
            key = (dataset, model, checkpoint, layer)
            score_index[key] = {
                **base,
                "ids": ids,
                "labels": labels,
                "scores": scores,
            }
            direction_index[key] = {
                **base,
                "unit_direction": unit_direction,
                "n_total": stats["n_total"],
            }

    model_agreement_records = []
    grouped_score_keys = {}
    for dataset, model, checkpoint, layer in score_index:
        grouped_score_keys.setdefault((dataset, checkpoint), []).append((model, layer))

    for (dataset, checkpoint), model_layers in tqdm(
        sorted(grouped_score_keys.items()),
        desc="Preference model agreement",
    ):
        models = sort_names_with_architecture_hint({model for model, _ in model_layers})
        for model_a, model_b in combinations(models, 2):
            layers_a = sorted(layer for model, layer in model_layers if model == model_a)
            layers_b = sorted(layer for model, layer in model_layers if model == model_b)
            for layer_a in layers_a:
                entry_a = score_index[(dataset, model_a, checkpoint, layer_a)]
                for layer_b in layers_b:
                    entry_b = score_index[(dataset, model_b, checkpoint, layer_b)]
                    idx_a, idx_b = align_ids(entry_a["ids"], entry_b["ids"])
                    if len(idx_a) < 2:
                        continue
                    scores_a = entry_a["scores"][idx_a]
                    scores_b = entry_b["scores"][idx_b]
                    model_agreement_records.append(
                        {
                            "split": entry_a["split"],
                            "dataset": dataset,
                            "source_datasets": entry_a["source_datasets"],
                            "checkpoint": checkpoint,
                            "checkpoint_step": checkpoint_step(checkpoint),
                            "model_a": model_a,
                            "model_b": model_b,
                            "layer_a": layer_a,
                            "layer_b": layer_b,
                            "layer_depth_a": entry_a["layer_depth"],
                            "layer_depth_b": entry_b["layer_depth"],
                            "n_aligned": int(len(idx_a)),
                            "score_pearson": pearson_correlation(scores_a, scores_b),
                            "score_centered_cosine": pearson_correlation(scores_a, scores_b),
                            "is_aggregate": entry_a["is_aggregate"],
                            "n_folds": entry_a["n_folds"],
                        }
                    )

    task_agreement_records = []
    grouped_direction_keys = {}
    for dataset, model, checkpoint, layer in direction_index:
        grouped_direction_keys.setdefault((model, checkpoint, layer), []).append(dataset)

    for (model, checkpoint, layer), datasets in tqdm(
        sorted(grouped_direction_keys.items()),
        desc="Preference task agreement",
    ):
        datasets = sorted(set(datasets))
        if len(datasets) < 2:
            continue
        first_entry = direction_index[(datasets[0], model, checkpoint, layer)]
        for dataset_a, dataset_b in combinations(datasets, 2):
            entry_a = direction_index[(dataset_a, model, checkpoint, layer)]
            entry_b = direction_index[(dataset_b, model, checkpoint, layer)]
            cosine = vector_cosine_similarity(entry_a["unit_direction"], entry_b["unit_direction"])
            task_agreement_records.append(
                {
                    "split": entry_a["split"],
                    "model": model,
                    "checkpoint": checkpoint,
                    "checkpoint_step": checkpoint_step(checkpoint),
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "source_datasets_a": entry_a["source_datasets"],
                    "source_datasets_b": entry_b["source_datasets"],
                    "layer": layer,
                    "layer_depth": first_entry["layer_depth"],
                    "direction_cosine": cosine,
                    "direction_abs_cosine": abs(cosine) if np.isfinite(cosine) else math.nan,
                    "n_total_a": entry_a["n_total"],
                    "n_total_b": entry_b["n_total"],
                    "is_aggregate_a": entry_a["is_aggregate"],
                    "is_aggregate_b": entry_b["is_aggregate"],
                    "n_folds_a": entry_a["n_folds"],
                    "n_folds_b": entry_b["n_folds"],
                }
            )

    return (
        pd.DataFrame(direction_records),
        pd.DataFrame(model_agreement_records),
        pd.DataFrame(task_agreement_records),
    )


def plot_preference_direction_strength(direction_df: pd.DataFrame, plots_dir: Path) -> None:
    if direction_df.empty:
        return
    pref_dir = plots_dir / "preference_directions"
    pref_dir.mkdir(parents=True, exist_ok=True)
    direction_df = drop_temporal_fold_rows(direction_df)
    if direction_df.empty:
        return

    for checkpoint, checkpoint_df in direction_df.groupby("checkpoint", dropna=False):
        dataset_names = sorted(checkpoint_df["dataset"].unique().tolist())
        models = sort_names_with_architecture_hint(checkpoint_df["model"].unique().tolist())
        palette = sns.color_palette("tab10", n_colors=len(models))
        color_map = dict(zip(models, palette))

        fig, axes = plt.subplots(nrows=1, ncols=len(dataset_names), figsize=(5.4 * len(dataset_names), 4.2), sharey=True)
        axes = [axes] if len(dataset_names) == 1 else list(axes)
        for ax, dataset in zip(axes, dataset_names):
            subset = checkpoint_df[checkpoint_df["dataset"] == dataset]
            for model in models:
                model_data = subset[subset["model"] == model].sort_values("layer_depth")
                if model_data.empty:
                    continue
                ax.plot(
                    model_data["layer_depth"],
                    model_data["score_cohens_d"],
                    color=color_map[model],
                    marker="o",
                    linewidth=1.9,
                    markersize=3.5,
                    alpha=0.9,
                    label=short_model_name(model),
                )
            style_plain_ax(ax, "Relative depth", "Direction strength (Cohen d)")
            ax.set_xlim(-0.02, 1.02)
            ax.set_title(dataset, fontsize=13)

        handles = [
            plt.Line2D(
                [0],
                [0],
                color=color_map[model],
                marker="o",
                linewidth=1.9,
                markersize=3.5,
                label=short_model_name(model),
            )
            for model in models
        ]
        fig.suptitle(f"Preference direction strength — {checkpoint}", fontsize=16)
        fig.legend(handles, [handle.get_label() for handle in handles], loc="lower center",
                   ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.08, 1, 0.94])
        out = pref_dir / f"geometry_preference_direction_strength_{checkpoint_plot_label(checkpoint)}.svg"
        fig.savefig(out, dpi=220)
        plt.close(fig)


def plot_preference_model_agreement(model_df: pd.DataFrame, plots_dir: Path) -> None:
    if model_df.empty:
        return
    pref_dir = plots_dir / "preference_directions" / "model_agreement"
    pref_dir.mkdir(parents=True, exist_ok=True)
    model_df = drop_temporal_fold_rows(model_df)
    final_df = model_df[
        np.isclose(model_df["layer_depth_a"], 1.0)
        & np.isclose(model_df["layer_depth_b"], 1.0)
    ].copy()
    if final_df.empty:
        return

    for (dataset, checkpoint), subset in final_df.groupby(["dataset", "checkpoint"], dropna=False):
        models = sort_names_with_architecture_hint(
            sorted(set(subset["model_a"].unique()).union(set(subset["model_b"].unique())))
        )
        labels = [short_model_name(model) for model in models]
        matrix = pd.DataFrame(np.eye(len(models)), index=labels, columns=labels, dtype=float)
        for _, row in subset.iterrows():
            a = short_model_name(row["model_a"])
            b = short_model_name(row["model_b"])
            matrix.loc[a, b] = row["score_pearson"]
            matrix.loc[b, a] = row["score_pearson"]

        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="vlag",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            cbar_kws={"label": "Preference-score correlation"},
        )
        ax.set_title(f"Model agreement on preference direction — {dataset} ({checkpoint})", fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("")
        fig.tight_layout()
        out = pref_dir / (
            f"geometry_preference_model_agreement_{safe_name(dataset)}_"
            f"{checkpoint_plot_label(checkpoint)}.svg"
        )
        fig.savefig(out, dpi=220)
        plt.close(fig)


def plot_preference_task_agreement(task_df: pd.DataFrame, plots_dir: Path) -> None:
    if task_df.empty:
        return
    pref_dir = plots_dir / "preference_directions" / "task_agreement"
    pref_dir.mkdir(parents=True, exist_ok=True)
    task_df = task_df[
        ~task_df["dataset_a"].astype(str).str.startswith("temporal_concord_fold_")
        & ~task_df["dataset_b"].astype(str).str.startswith("temporal_concord_fold_")
    ].copy()
    final_df = task_df[np.isclose(task_df["layer_depth"], 1.0)].copy()
    if final_df.empty:
        return

    for (model, checkpoint), subset in final_df.groupby(["model", "checkpoint"], dropna=False):
        datasets = sorted(set(subset["dataset_a"].unique()).union(set(subset["dataset_b"].unique())))
        matrix = pd.DataFrame(np.eye(len(datasets)), index=datasets, columns=datasets, dtype=float)
        for _, row in subset.iterrows():
            matrix.loc[row["dataset_a"], row["dataset_b"]] = row["direction_cosine"]
            matrix.loc[row["dataset_b"], row["dataset_a"]] = row["direction_cosine"]

        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="vlag",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            cbar_kws={"label": "Direction cosine"},
        )
        ax.set_title(
            f"Task preference-direction agreement — {short_model_name(model)} ({checkpoint})",
            fontsize=13,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        fig.tight_layout()
        out = pref_dir / (
            f"geometry_preference_task_agreement_{safe_name(short_model_name(model))}_"
            f"{checkpoint_plot_label(checkpoint)}.svg"
        )
        fig.savefig(out, dpi=220)
        plt.close(fig)


def plot_preference_direction_analysis(
    direction_df: pd.DataFrame,
    model_agreement_df: pd.DataFrame,
    task_agreement_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    plot_preference_direction_strength(direction_df, plots_dir)
    plot_preference_model_agreement(model_agreement_df, plots_dir)
    plot_preference_task_agreement(task_agreement_df, plots_dir)


def plot_layer_trajectories(class_df: pd.DataFrame, id_df: pd.DataFrame, plots_dir: Path) -> None:
    if class_df.empty:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    class_df = drop_temporal_fold_rows(class_df)
    id_df = drop_temporal_fold_rows(id_df)
    if class_df.empty:
        return
    grouped = class_df.groupby(["dataset", "model", "checkpoint"], dropna=False)
    palette = sns.color_palette("tab10", n_colors=3)

    for (dataset, model, checkpoint), class_subset in grouped:
        id_subset = id_df[
            (id_df["dataset"] == dataset)
            & (id_df["model"] == model)
            & (id_df["checkpoint"] == checkpoint)
        ]
        if class_subset.empty or id_subset.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
        axes = axes.ravel()

        class_subset = class_subset.sort_values("layer_depth")
        # Balanced binary datasets put the pooled centroid exactly between class centroids.
        # Different styles keep class_0 visible when class_0 and class_1 distances overlap.
        class_line_styles = {
            "between": {
                "color": palette[0],
                "linestyle": "-",
                "marker": "o",
                "linewidth": 1.9,
                "markersize": 3.5,
                "alpha": 0.9,
                "zorder": 3,
            },
            "class_0": {
                "color": palette[1],
                "linestyle": "-",
                "marker": "o",
                "linewidth": 1.9,
                "markersize": 3.5,
                "alpha": 0.9,
                "zorder": 1,
            },
            "class_1": {
                "color": palette[2],
                "linestyle": "--",
                "marker": "s",
                "linewidth": 1.9,
                "markersize": 3.5,
                "alpha": 0.9,
                "zorder": 2,
            },
        }
        centroid_series = {
            "between": "centroid_euclidean",
            "class_0": "centroid_euclidean_class_0",
            "class_1": "centroid_euclidean_class_1",
        }
        for label, column in centroid_series.items():
            if column not in class_subset.columns:
                continue
            axes[0].plot(
                class_subset["layer_depth"],
                class_subset[column],
                label=label,
                **class_line_styles[label],
            )
        axes[0].set_title("Centroid distance")
        style_plain_ax(axes[0], "Relative depth", "Euclidean")

        fisher_series = {
            "between": "fisher_ratio",
            "class_0": "fisher_ratio_class_0",
            "class_1": "fisher_ratio_class_1",
        }
        for label, column in fisher_series.items():
            if column not in class_subset.columns:
                continue
            axes[1].plot(
                class_subset["layer_depth"],
                class_subset[column],
                label=label,
                **class_line_styles[label],
            )
        axes[1].set_title("Fisher ratio")
        style_plain_ax(axes[1], "Relative depth", "Ratio")

        subset_styles = {
            "all": class_line_styles["between"],
            "class_0": class_line_styles["class_0"],
            "class_1": class_line_styles["class_1"],
        }
        for subset_name in ["all", "class_0", "class_1"]:
            sub = id_subset[id_subset["subset"] == subset_name].sort_values("layer_depth")
            if sub.empty:
                continue
            axes[2].plot(
                sub["layer_depth"],
                sub["pca_rank_99"],
                label=subset_name,
                **subset_styles[subset_name],
            )
            axes[3].plot(
                sub["layer_depth"],
                sub["ess_id"],
                label=subset_name,
                **subset_styles[subset_name],
            )

        axes[2].set_title("PCA 99% rank")
        style_plain_ax(axes[2], "Relative depth", "Components")
        axes[3].set_title("ESS ID")
        style_plain_ax(axes[3], "Relative depth", "ID")
        for ax in axes:
            ax.set_xlim(-0.02, 1.02)
        axes[0].legend(frameon=False, fontsize=9)
        axes[1].legend(frameon=False, fontsize=9)
        axes[2].legend(frameon=False, fontsize=9)
        axes[3].legend(frameon=False, fontsize=9)

        fig.suptitle(
            f"Representation geometry — {dataset} — {short_model_name(model)} ({checkpoint})",
            fontsize=14,
        )
        fig.tight_layout()
        out = plots_dir / (
            f"geometry_layer_profile_{safe_name(dataset)}_"
            f"{safe_name(short_model_name(model))}_{checkpoint_plot_label(checkpoint)}.svg"
        )
        fig.savefig(out, dpi=220)
        plt.close(fig)


def plot_model_comparison_metric(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    output_prefix: str,
    plots_dir: Path,
    style_col: Optional[str] = None,
    style_order: Optional[Sequence[str]] = None,
) -> None:
    if df.empty or metric_col not in df.columns:
        return
    df = drop_temporal_fold_rows(df)
    df = df[np.isfinite(df[metric_col])].copy()
    if df.empty:
        return

    comparison_dir = plots_dir / "model_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint, checkpoint_df in df.groupby("checkpoint", dropna=False):
        dataset_names = sorted(checkpoint_df["dataset"].unique().tolist())
        models = sort_names_with_architecture_hint(checkpoint_df["model"].unique().tolist())
        if not dataset_names or not models:
            continue
        style_values: List[str] = []
        if style_col is not None and style_col in checkpoint_df.columns:
            present_styles = set(checkpoint_df[style_col].dropna().unique().tolist())
            style_values = [
                value
                for value in (style_order or sorted(present_styles))
                if value in present_styles
            ]

        palette = sns.color_palette("tab10", n_colors=len(models))
        color_map = dict(zip(models, palette))
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(dataset_names),
            figsize=(5.4 * len(dataset_names), 4.2),
            sharey=False,
        )
        axes = [axes] if len(dataset_names) == 1 else list(axes)

        for ax, dataset in zip(axes, dataset_names):
            subset = checkpoint_df[checkpoint_df["dataset"] == dataset]
            for model in models:
                if style_values:
                    for style_value in style_values:
                        model_data = subset[
                            (subset["model"] == model)
                            & (subset[style_col] == style_value)
                        ].sort_values("layer_depth")
                        if model_data.empty:
                            continue
                        style = CLASS_SUBSET_STYLES.get(style_value, {})
                        ax.plot(
                            model_data["layer_depth"],
                            model_data[metric_col],
                            color=color_map[model],
                            linewidth=1.9,
                            markersize=3.5,
                            label=short_model_name(model),
                            **style,
                        )
                else:
                    model_data = subset[subset["model"] == model].sort_values("layer_depth")
                    if model_data.empty:
                        continue
                    ax.plot(
                        model_data["layer_depth"],
                        model_data[metric_col],
                        color=color_map[model],
                        marker="o",
                        linewidth=1.9,
                        markersize=3.5,
                        alpha=0.9,
                        label=short_model_name(model),
                    )
            style_plain_ax(ax, "Relative depth", ylabel)
            ax.set_xlim(-0.02, 1.02)
            ax.set_title(dataset, fontsize=13)

        handles = [
            plt.Line2D(
                [0],
                [0],
                color=color_map[model],
                marker="o",
                linewidth=1.9,
                markersize=3.5,
                label=short_model_name(model),
            )
            for model in models
        ]
        if style_values:
            handles.extend(
                plt.Line2D(
                    [0],
                    [0],
                    color="0.25",
                    linewidth=1.9,
                    markersize=3.5,
                    label=CLASS_SUBSET_LABELS.get(style_value, style_value),
                    **CLASS_SUBSET_STYLES.get(style_value, {}),
                )
                for style_value in style_values
            )
        fig.suptitle(f"{title} — {checkpoint}", fontsize=16)
        fig.legend(
            handles,
            [handle.get_label() for handle in handles],
            loc="lower center",
            ncol=min(5, len(handles)),
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.tight_layout(rect=[0, 0.08, 1, 0.94])
        out = comparison_dir / f"{output_prefix}_{checkpoint_plot_label(checkpoint)}.svg"
        fig.savefig(out, dpi=220)
        plt.close(fig)


def class_metric_frame(df: pd.DataFrame, metric_columns: Dict[str, str]) -> pd.DataFrame:
    frames = []
    for class_subset, metric_col in metric_columns.items():
        if metric_col not in df.columns:
            continue
        sub = df.copy()
        sub["class_subset"] = class_subset
        sub["metric_value"] = sub[metric_col]
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_model_comparisons(
    class_df: pd.DataFrame,
    id_df: pd.DataFrame,
    plots_dir: Path,
    preference_df: Optional[pd.DataFrame] = None,
) -> None:
    centroid_by_class = class_metric_frame(
        class_df,
        {
            "class_0": "centroid_euclidean_class_0",
            "class_1": "centroid_euclidean_class_1",
        },
    )
    plot_model_comparison_metric(
        centroid_by_class,
        metric_col="metric_value",
        ylabel="Euclidean",
        title="Model comparison: centroid distance by class",
        output_prefix="geometry_model_comparison_centroid_distance",
        plots_dir=plots_dir,
        style_col="class_subset",
        style_order=["class_0", "class_1"],
    )
    fisher_by_class = class_metric_frame(
        class_df,
        {
            "class_0": "fisher_ratio_class_0",
            "class_1": "fisher_ratio_class_1",
        },
    )
    plot_model_comparison_metric(
        fisher_by_class,
        metric_col="metric_value",
        ylabel="Ratio",
        title="Model comparison: Fisher ratio by class",
        output_prefix="geometry_model_comparison_fisher_ratio",
        plots_dir=plots_dir,
        style_col="class_subset",
        style_order=["class_0", "class_1"],
    )

    separation_df = class_df.copy()
    if {
        "between_class_mean_cosine_distance",
        "within_class_0_mean_cosine_distance",
        "within_class_1_mean_cosine_distance",
    }.issubset(separation_df.columns):
        separation_df["separation_ratio_class_0"] = (
            separation_df["between_class_mean_cosine_distance"]
            / (separation_df["within_class_0_mean_cosine_distance"] + EPS)
        )
        separation_df["separation_ratio_class_1"] = (
            separation_df["between_class_mean_cosine_distance"]
            / (separation_df["within_class_1_mean_cosine_distance"] + EPS)
        )
    separation_by_class = class_metric_frame(
        separation_df,
        {
            "class_0": "separation_ratio_class_0",
            "class_1": "separation_ratio_class_1",
        },
    )
    plot_model_comparison_metric(
        separation_by_class,
        metric_col="metric_value",
        ylabel="Between / within cosine distance",
        title="Model comparison: separation ratio by class",
        output_prefix="geometry_model_comparison_separation_ratio",
        plots_dir=plots_dir,
        style_col="class_subset",
        style_order=["class_0", "class_1"],
    )

    if not id_df.empty:
        id_classes = id_df[id_df["subset"].isin(["class_0", "class_1"])].copy()
        plot_model_comparison_metric(
            id_classes,
            metric_col="pca_rank_99",
            ylabel="Components",
            title="Model comparison: PCA 99% rank by class",
            output_prefix="geometry_model_comparison_pca99_rank",
            plots_dir=plots_dir,
            style_col="subset",
            style_order=["class_0", "class_1"],
        )
        plot_model_comparison_metric(
            id_classes,
            metric_col="ess_id",
            ylabel="ID",
            title="Model comparison: ESS ID by class",
            output_prefix="geometry_model_comparison_ess_id",
            plots_dir=plots_dir,
            style_col="subset",
            style_order=["class_0", "class_1"],
        )

    if preference_df is not None and not preference_df.empty:
        preference_by_class = class_metric_frame(
            preference_df,
            {
                "class_0": "score_mean_class_0",
                "class_1": "score_mean_class_1",
            },
        )
        plot_model_comparison_metric(
            preference_by_class,
            metric_col="metric_value",
            ylabel="Mean direction score",
            title="Model comparison: preference direction score by class",
            output_prefix="geometry_model_comparison_preference_direction_strength",
            plots_dir=plots_dir,
            style_col="class_subset",
            style_order=["class_0", "class_1"],
        )


def plot_knn_model_comparison(knn_df: pd.DataFrame, plots_dir: Path) -> None:
    if knn_df.empty:
        return
    knn_df = drop_temporal_fold_rows(knn_df)
    if knn_df.empty:
        return
    plot_k = int(knn_df["k"].dropna().iloc[0]) if "k" in knn_df.columns and knn_df["k"].notna().any() else KNN_NEIGHBORS

    base_cols = ["split", "dataset", "checkpoint", "checkpoint_step", "class_subset", "mean_overlap_fraction"]
    left = knn_df[base_cols].copy()
    left["model"] = knn_df["model_a"]
    left["layer"] = knn_df["layer_a"]
    left["layer_depth"] = knn_df["layer_depth_a"]
    right = knn_df[base_cols].copy()
    right["model"] = knn_df["model_b"]
    right["layer"] = knn_df["layer_b"]
    right["layer_depth"] = knn_df["layer_depth_b"]
    model_df = pd.concat([left, right], ignore_index=True, sort=False)
    group_cols = ["split", "dataset", "checkpoint", "checkpoint_step", "model", "layer", "layer_depth", "class_subset"]
    group_cols = [col for col in group_cols if col in model_df.columns]
    model_df = (
        model_df.groupby(group_cols, dropna=False)["mean_overlap_fraction"]
        .mean()
        .reset_index()
    )
    plot_model_comparison_metric(
        model_df,
        metric_col="mean_overlap_fraction",
        ylabel=f"Shared {plot_k}-NN fraction",
        title="Model comparison: neighborhood agreement by class",
        output_prefix=f"geometry_model_comparison_knn_neighborhood_agreement_k{plot_k}",
        plots_dir=plots_dir,
        style_col="class_subset",
        style_order=["class_0", "class_1"],
    )


def plot_knn_pair_heatmaps(knn_df: pd.DataFrame, plots_dir: Path) -> None:
    if knn_df.empty:
        return
    knn_df = drop_temporal_fold_rows(knn_df)
    if knn_df.empty:
        return

    out_dir = plots_dir / "neighborhood_agreement"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = sns.color_palette("crest", as_cmap=True)

    for (checkpoint, dataset, class_subset), subset in knn_df.groupby(
        ["checkpoint", "dataset", "class_subset"], dropna=False
    ):
        plot_k = int(subset["k"].dropna().iloc[0]) if "k" in subset.columns and subset["k"].notna().any() else KNN_NEIGHBORS
        models = sort_names_with_architecture_hint(
            sorted(set(subset["model_a"].tolist()) | set(subset["model_b"].tolist()))
        )
        if len(models) < 2:
            continue

        matrix = pd.DataFrame(np.eye(len(models)), index=models, columns=models, dtype=float)
        pair_values = (
            subset.groupby(["model_a", "model_b"], dropna=False)["mean_overlap_fraction"]
            .mean()
            .reset_index()
        )
        for _, row in pair_values.iterrows():
            model_a = row["model_a"]
            model_b = row["model_b"]
            value = row["mean_overlap_fraction"]
            matrix.loc[model_a, model_b] = value
            matrix.loc[model_b, model_a] = value

        display = matrix.copy()
        display.index = [short_model_name(model) for model in display.index]
        display.columns = [short_model_name(model) for model in display.columns]
        fig, ax = plt.subplots(figsize=(7.4, 6.2))
        sns.heatmap(
            display,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": f"Shared {plot_k}-NN fraction"},
        )
        class_label = CLASS_SUBSET_LABELS.get(class_subset, class_subset)
        ax.set_title(
            f"{dataset}: neighborhood agreement ({class_label}) — {checkpoint}",
            fontsize=13,
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("Model")
        fig.tight_layout()
        out = out_dir / (
            f"geometry_knn_agreement_matrix_{safe_name(dataset)}_"
            f"{safe_name(class_subset)}_{checkpoint_plot_label(checkpoint)}_k{plot_k}.svg"
        )
        fig.savefig(out, dpi=220)
        plt.close(fig)


def plot_knn_neighborhood_agreement(knn_df: pd.DataFrame, plots_dir: Path) -> None:
    plot_knn_model_comparison(knn_df, plots_dir)
    plot_knn_pair_heatmaps(knn_df, plots_dir)


def plot_cka_heatmaps(cka_df: pd.DataFrame, plots_dir: Path) -> None:
    if cka_df.empty:
        return
    cka_df = drop_temporal_fold_rows(cka_df)
    if cka_df.empty:
        return
    heatmap_dir = plots_dir / "cka_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    group_cols = ["dataset", "checkpoint_a", "checkpoint_b", "model_a", "model_b"]
    cmap = sns.color_palette("crest", as_cmap=True)

    for key, subset in cka_df.groupby(group_cols, dropna=False):
        dataset, checkpoint_a, checkpoint_b, model_a, model_b = key
        pivot = subset.pivot_table(index="layer_a", columns="layer_b", values="cka", aggfunc="mean")
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        sns.heatmap(pivot, ax=ax, cmap=cmap, vmin=0, vmax=1, cbar_kws={"label": "Linear CKA"})
        ax.set_title(
            f"{dataset}: {short_model_name(model_a)} vs {short_model_name(model_b)}",
            fontsize=13,
        )
        ax.set_xlabel(f"{short_model_name(model_b)} layer")
        ax.set_ylabel(f"{short_model_name(model_a)} layer")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        out = heatmap_dir / (
            f"geometry_cka_heatmap_{safe_name(dataset)}_{safe_name(short_model_name(model_a))}_"
            f"{safe_name(short_model_name(model_b))}_{checkpoint_plot_label(checkpoint_a)}_"
            f"{checkpoint_plot_label(checkpoint_b)}.svg"
        )
        fig.savefig(out, dpi=220)
        plt.close(fig)


def plot_checkpoint_drift(drift_df: pd.DataFrame, plots_dir: Path) -> None:
    if drift_df.empty:
        return
    drift_df = drop_temporal_fold_rows(drift_df)
    if drift_df.empty:
        return
    drift_dir = plots_dir / "checkpoint_drift"
    drift_dir.mkdir(parents=True, exist_ok=True)

    cka_df = drift_df[drift_df["metric"] == "cka_same_layer"]
    for key, subset in cka_df.groupby(["dataset", "model", "checkpoint_a", "checkpoint_b"], dropna=False):
        dataset, model, checkpoint_a, checkpoint_b = key
        subset = subset.sort_values("layer_depth")
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(subset["layer_depth"], subset["cka"], marker="o", linewidth=1.9, markersize=3.5, alpha=0.9)
        ax.set_title(f"{dataset} — {short_model_name(model)}", fontsize=13)
        style_plain_ax(ax, "Relative depth", "Same-layer CKA")
        ax.set_xlim(-0.02, 1.02)
        fig.tight_layout()
        out = drift_dir / (
            f"geometry_checkpoint_cka_{safe_name(dataset)}_{safe_name(short_model_name(model))}_"
            f"{checkpoint_plot_label(checkpoint_a)}_{checkpoint_plot_label(checkpoint_b)}.svg"
        )
        fig.savefig(out, dpi=220)
        plt.close(fig)

    centroid_df = drift_df[drift_df["metric"] == "centroid_drift_to_reference"]
    for key, subset in centroid_df.groupby(["dataset", "model", "reference_checkpoint"], dropna=False):
        dataset, model, reference = key
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for (checkpoint, label), sub in subset.groupby(["checkpoint_a", "label"], dropna=False):
            sub = sub.sort_values("layer_depth")
            ax.plot(
                sub["layer_depth"],
                sub["centroid_drift_euclidean"],
                marker="o",
                linewidth=1.9,
                markersize=3.5,
                alpha=0.9,
                label=f"{checkpoint} label={int(label)}" if np.isfinite(label) else str(checkpoint),
            )
        ax.set_title(f"{dataset} — {short_model_name(model)}", fontsize=13)
        style_plain_ax(ax, "Relative depth", "Euclidean drift")
        ax.set_xlim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        out = drift_dir / (
            f"geometry_centroid_drift_{safe_name(dataset)}_{safe_name(short_model_name(model))}_"
            f"to_{checkpoint_plot_label(reference)}.svg"
        )
        fig.savefig(out, dpi=220)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.pca2d_only or args.preference_only or args.neighborhood_only:
        args.plot = True
    np.random.seed(args.seed)
    if torch is not None:
        torch.manual_seed(args.seed)

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plots_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_from_results:
        class_path = results_dir / "geometry_class_separation.csv"
        id_path = results_dir / "geometry_intrinsic_dimensionality.csv"
        if not class_path.exists() or not id_path.exists():
            raise FileNotFoundError(
                "plot_from_results requires geometry_class_separation.csv and "
                "geometry_intrinsic_dimensionality.csv in results_dir."
            )
        class_df = pd.read_csv(class_path)
        id_df = pd.read_csv(id_path)
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_layer_trajectories(class_df, id_df, plots_dir)
        preference_direction_df = None
        if not args.skip_cka:
            cka_path = results_dir / "geometry_cka_cross_model.csv"
            if cka_path.exists():
                try:
                    cka_df = pd.read_csv(cka_path)
                    plot_cka_heatmaps(cka_df, plots_dir)
                except pd.errors.EmptyDataError:
                    warn(f"Skipping CKA plots because {cka_path} is empty.")
        drift_path = results_dir / "geometry_checkpoint_drift.csv"
        if drift_path.exists():
            try:
                drift_df = pd.read_csv(drift_path)
                plot_checkpoint_drift(drift_df, plots_dir)
            except pd.errors.EmptyDataError:
                warn(f"Skipping checkpoint drift plots because {drift_path} is empty.")
        preference_paths = [
            results_dir / "geometry_preference_directions.csv",
            results_dir / "geometry_preference_model_agreement.csv",
            results_dir / "geometry_preference_task_agreement.csv",
        ]
        if all(path.exists() for path in preference_paths):
            try:
                preference_direction_df = pd.read_csv(preference_paths[0])
                preference_model_df = pd.read_csv(preference_paths[1])
                preference_task_df = pd.read_csv(preference_paths[2])
                plot_preference_direction_analysis(
                    preference_direction_df,
                    preference_model_df,
                    preference_task_df,
                    plots_dir,
                )
            except pd.errors.EmptyDataError:
                warn("Skipping preference-direction plots because one preference CSV is empty.")
        plot_model_comparisons(class_df, id_df, plots_dir, preference_direction_df)
        knn_path = results_dir / "geometry_knn_neighborhood_agreement.csv"
        if knn_path.exists():
            try:
                knn_df = pd.read_csv(knn_path)
                plot_knn_neighborhood_agreement(knn_df, plots_dir)
            except pd.errors.EmptyDataError:
                warn(f"Skipping kNN neighborhood plots because {knn_path} is empty.")
        return

    dataset_specs = discover_datasets(Path(args.test_dir), args.datasets)
    datasets = {
        spec.name: load_dataset(spec, max_samples=args.max_samples, seed=args.seed)
        for spec in dataset_specs
    }

    model_names = discover_model_names(Path(args.models_dir), args.model_names)
    checkpoint_labels = requested_checkpoint_labels(args)
    if args.checkpoints is None:
        args.include_final = True

    representation_index = ensure_representation_caches(
        args=args,
        model_names=model_names,
        checkpoint_labels=checkpoint_labels,
        dataset_specs=dataset_specs,
        datasets=datasets,
    )

    if not representation_index:
        warn("No representation caches are available; writing empty result CSVs.")
        write_csv(pd.DataFrame(), results_dir / "geometry_class_separation.csv", CLASS_COLUMNS)
        write_csv(pd.DataFrame(), results_dir / "geometry_intrinsic_dimensionality.csv", ID_COLUMNS)
        write_csv(pd.DataFrame(), results_dir / "geometry_cka_cross_model.csv", CKA_COLUMNS)
        write_csv(pd.DataFrame(), results_dir / "geometry_pca2d_variance.csv", PCA2D_COLUMNS)
        write_csv(pd.DataFrame(), results_dir / "geometry_preference_directions.csv", PREFERENCE_DIRECTION_COLUMNS)
        write_csv(pd.DataFrame(), results_dir / "geometry_preference_model_agreement.csv", PREFERENCE_MODEL_AGREEMENT_COLUMNS)
        write_csv(pd.DataFrame(), results_dir / "geometry_preference_task_agreement.csv", PREFERENCE_TASK_AGREEMENT_COLUMNS)
        write_csv(pd.DataFrame(), results_dir / "geometry_knn_neighborhood_agreement.csv", KNN_AGREEMENT_COLUMNS)
        return

    if args.neighborhood_only:
        knn_df = compute_knn_neighborhood_agreement(
            representation_index=representation_index,
            requested_layers_arg=args.layers,
            knn_neighbors=args.knn_neighbors,
        )
        write_csv(knn_df, results_dir / "geometry_knn_neighborhood_agreement.csv", KNN_AGREEMENT_COLUMNS)
        plot_knn_neighborhood_agreement(knn_df, plots_dir)
        return

    if args.preference_only:
        preference_direction_df, preference_model_df, preference_task_df = compute_preference_direction_analysis(
            representation_index=representation_index,
            dataset_specs=dataset_specs,
            requested_layers_arg=args.preference_layers if args.preference_layers is not None else args.layers,
        )
        write_csv(preference_direction_df, results_dir / "geometry_preference_directions.csv", PREFERENCE_DIRECTION_COLUMNS)
        write_csv(
            preference_model_df,
            results_dir / "geometry_preference_model_agreement.csv",
            PREFERENCE_MODEL_AGREEMENT_COLUMNS,
        )
        write_csv(
            preference_task_df,
            results_dir / "geometry_preference_task_agreement.csv",
            PREFERENCE_TASK_AGREEMENT_COLUMNS,
        )
        plot_preference_direction_analysis(
            preference_direction_df,
            preference_model_df,
            preference_task_df,
            plots_dir,
        )
        return

    if args.pca2d_only:
        pca2d_df = plot_pca2d_for_representations(
            representation_index=representation_index,
            dataset_specs=dataset_specs,
            requested_layers_arg=args.layers,
            requested_pca_layers_arg=args.pca2d_layers,
            max_samples=args.pca2d_max_samples,
            seed=args.seed,
            plots_dir=plots_dir,
        )
        write_csv(pca2d_df, results_dir / "geometry_pca2d_variance.csv", PCA2D_COLUMNS)
        return

    class_df, id_df = compute_layer_metrics(
        representation_index=representation_index,
        dataset_specs=dataset_specs,
        requested_layers_arg=args.layers,
        seed=args.seed,
    )
    write_csv(class_df, results_dir / "geometry_class_separation.csv", CLASS_COLUMNS)
    write_csv(id_df, results_dir / "geometry_intrinsic_dimensionality.csv", ID_COLUMNS)

    cka_df = pd.DataFrame()
    if not args.skip_cka:
        cka_df = compute_cross_model_cka(
            representation_index=representation_index,
            requested_layers_arg=args.layers,
        )
        write_csv(cka_df, results_dir / "geometry_cka_cross_model.csv", CKA_COLUMNS)

    checkpoint_counts = pd.Series([key[2] for key in representation_index]).nunique()
    drift_df = pd.DataFrame()
    if checkpoint_counts > 1:
        drift_df = compute_checkpoint_drift(
            representation_index=representation_index,
            requested_layers_arg=args.layers,
        )
        write_csv(drift_df, results_dir / "geometry_checkpoint_drift.csv", DRIFT_COLUMNS)

    preference_direction_df = pd.DataFrame()
    preference_model_df = pd.DataFrame()
    preference_task_df = pd.DataFrame()
    if not args.skip_preference:
        preference_direction_df, preference_model_df, preference_task_df = compute_preference_direction_analysis(
            representation_index=representation_index,
            dataset_specs=dataset_specs,
            requested_layers_arg=args.preference_layers if args.preference_layers is not None else args.layers,
        )
        write_csv(preference_direction_df, results_dir / "geometry_preference_directions.csv", PREFERENCE_DIRECTION_COLUMNS)
        write_csv(
            preference_model_df,
            results_dir / "geometry_preference_model_agreement.csv",
            PREFERENCE_MODEL_AGREEMENT_COLUMNS,
        )
        write_csv(
            preference_task_df,
            results_dir / "geometry_preference_task_agreement.csv",
            PREFERENCE_TASK_AGREEMENT_COLUMNS,
        )

    if args.plot:
        plot_layer_trajectories(class_df, id_df, plots_dir)
        plot_model_comparisons(
            class_df,
            id_df,
            plots_dir,
            preference_direction_df if not args.skip_preference else None,
        )
        plot_cka_heatmaps(cka_df, plots_dir)
        if not args.skip_preference:
            plot_preference_direction_analysis(
                preference_direction_df,
                preference_model_df,
                preference_task_df,
                plots_dir,
            )
        if not args.skip_pca2d:
            pca2d_df = plot_pca2d_for_representations(
                representation_index=representation_index,
                dataset_specs=dataset_specs,
                requested_layers_arg=args.layers,
                requested_pca_layers_arg=args.pca2d_layers,
                max_samples=args.pca2d_max_samples,
                seed=args.seed,
                plots_dir=plots_dir,
            )
            write_csv(pca2d_df, results_dir / "geometry_pca2d_variance.csv", PCA2D_COLUMNS)
        if not drift_df.empty:
            plot_checkpoint_drift(drift_df, plots_dir)


if __name__ == "__main__":
    main()
