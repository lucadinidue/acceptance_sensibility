"""Linear probes on task directions and variance-augmented task spaces.

This script asks whether acceptability can be decoded from a small number of
representation dimensions instead of the full representation space. The first
dimension is the train-set class-centroid direction. Optional extra dimensions
are unsupervised PCA directions of the train representations after removing the
centroid direction, so labels only define the first axis.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import representation_geometry as rg


sns.set_theme(style="whitegrid", font_scale=0.9)

RESULT_COLUMNS = [
    "target_dataset",
    "source_dataset",
    "source_datasets",
    "model",
    "checkpoint",
    "checkpoint_step",
    "layer",
    "layer_depth",
    "feature_kind",
    "feature_dim",
    "source_direction_count",
    "variance_component_count",
    "residual_variance_explained",
    "direction_variance_explained",
    "train_n",
    "test_n",
    "accuracy",
    "f1",
    "train_accuracy",
    "train_f1",
    "target_direction_capture",
    "source_target_direction_cosine",
    "is_target_aggregate",
    "n_target_folds",
]


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser("Probe acceptability from task-direction subspaces.")
    parser.add_argument("--models_dir", type=str, default="models/pretrained")
    parser.add_argument("--tokenizer_path", type=str, default="models/tokenizer")
    parser.add_argument("--train_dir", type=str, default="data/acceptance_datasets/train")
    parser.add_argument("--test_dir", type=str, default="data/acceptance_datasets/test")
    parser.add_argument("--representations_dir", type=str, default="data/representations_geometry")
    parser.add_argument("--results_dir", type=str, default="data/results")
    parser.add_argument("--plots_dir", type=str, default="plots/probing/direction_subspace")
    parser.add_argument("--model_names", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--include_final", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--layers", nargs="*", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_subspace_dims",
        type=int,
        default=3,
        help="Number of residual PCA variance dimensions to add to the 1D task direction.",
    )
    parser.add_argument("--include_full_probe", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def discover_split_datasets(data_dir: Path, split: str, requested: Optional[Sequence[str]]) -> List[rg.DatasetSpec]:
    if not data_dir.exists():
        rg.warn(f"Dataset directory not found: {data_dir}")
        return []

    paths = {path.stem: path for path in sorted(data_dir.glob("*.jsonl"))}
    if requested is None:
        names = sorted(paths)
    else:
        names = []
        for name in requested:
            if name == "temporal_concord":
                fold_names = sorted(path_name for path_name in paths if path_name.startswith("temporal_concord_fold_"))
                if not fold_names:
                    rg.warn(f"No temporal_concord_fold_*.jsonl files found in {data_dir}.")
                names.extend(fold_names)
            elif name in paths:
                names.append(name)
            else:
                rg.warn(f"Requested dataset {name}.jsonl not found in {data_dir}.")

    seen = set()
    specs = []
    for name in names:
        if name in seen or name not in paths:
            continue
        seen.add(name)
        specs.append(rg.DatasetSpec(name=name, path=paths[name], split=split))
    return specs


def load_split_data(specs: Sequence[rg.DatasetSpec], max_samples: Optional[int], seed: int) -> Dict[str, pd.DataFrame]:
    return {
        spec.name: rg.load_dataset(
            spec,
            max_samples=max_samples,
            seed=rg.stable_seed(spec.split, spec.name, base_seed=seed),
        )
        for spec in specs
    }


def probe_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    probe = RidgeClassifier(class_weight="balanced", random_state=seed)
    probe.fit(X_train_scaled, y_train)
    train_pred = probe.predict(X_train_scaled)
    test_pred = probe.predict(X_test_scaled)
    return {
        "accuracy": accuracy_score(y_test, test_pred),
        "f1": f1_score(y_test, test_pred, zero_division=0),
        "train_accuracy": accuracy_score(y_train, train_pred),
        "train_f1": f1_score(y_train, train_pred, zero_division=0),
    }


def projection_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    basis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    train_mean = X_train.mean(axis=0, keepdims=True)
    return (X_train - train_mean) @ basis, (X_test - train_mean) @ basis


def direction_capture(basis: np.ndarray, target_unit_direction: np.ndarray) -> float:
    if basis.size == 0 or len(target_unit_direction) == 0:
        return math.nan
    return float(np.linalg.norm(basis.T @ target_unit_direction))


def direction_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return math.nan
    return rg.vector_cosine_similarity(a, b)


def direction_plus_variance_basis(
    X_train: np.ndarray,
    unit_direction: np.ndarray,
    n_variance_dims: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Build [centroid direction, residual PCA directions].

    The PCA directions are fit on centered train representations after removing
    the 1D centroid direction. This keeps the first axis interpretable and uses
    the remaining axes only for unsupervised high-variance structure.
    """
    unit_direction = np.asarray(unit_direction, dtype=np.float64)
    direction_norm = np.linalg.norm(unit_direction)
    if direction_norm <= rg.EPS:
        return np.zeros((0, 0), dtype=np.float64), {
            "variance_component_count": 0,
            "residual_variance_explained": math.nan,
            "direction_variance_explained": math.nan,
        }

    unit_direction = unit_direction / direction_norm
    X_train = np.asarray(X_train, dtype=np.float64)
    centered = X_train - X_train.mean(axis=0, keepdims=True)
    total_ss = float(np.sum(centered * centered))
    direction_scores = centered @ unit_direction
    direction_ss = float(np.sum(direction_scores * direction_scores))
    residual = centered - np.outer(direction_scores, unit_direction)
    residual_ss = float(np.sum(residual * residual))

    components = []
    explained = 0.0
    if n_variance_dims > 0 and residual_ss > rg.EPS:
        try:
            covariance = residual.T @ residual
            eigvals, eigvecs = np.linalg.eigh(covariance)
            order = np.argsort(eigvals)[::-1]
        except np.linalg.LinAlgError:
            order = []
            eigvals = np.asarray([], dtype=np.float64)
            eigvecs = np.zeros((residual.shape[1], 0), dtype=np.float64)

        for idx in order:
            if len(components) >= n_variance_dims:
                break
            if eigvals[idx] <= rg.EPS:
                continue
            candidate = eigvecs[:, idx].astype(np.float64)
            candidate = candidate - unit_direction * float(candidate @ unit_direction)
            for previous in components:
                candidate = candidate - previous * float(candidate @ previous)
            norm = np.linalg.norm(candidate)
            if norm <= rg.EPS:
                continue
            components.append(candidate / norm)
            explained += float(eigvals[idx])

    vectors = [unit_direction] + components
    basis = np.column_stack(vectors)
    return basis, {
        "variance_component_count": len(components),
        "residual_variance_explained": explained / residual_ss if residual_ss > rg.EPS else math.nan,
        "direction_variance_explained": direction_ss / total_ss if total_ss > rg.EPS else math.nan,
    }


def direction_record_base(
    target_dataset: str,
    source_dataset: str,
    source_datasets: Sequence[str],
    model: str,
    checkpoint: str,
    layer: int,
    layer_depth: float,
    feature_kind: str,
    feature_dim: int,
    source_direction_count: int,
    train_n: int,
    test_n: int,
    target_direction_capture: float,
    source_target_direction_cosine: float,
    is_target_aggregate: bool,
    n_target_folds: float,
    variance_component_count: float = math.nan,
    residual_variance_explained: float = math.nan,
    direction_variance_explained: float = math.nan,
) -> Dict[str, object]:
    return {
        "target_dataset": target_dataset,
        "source_dataset": source_dataset,
        "source_datasets": ";".join(source_datasets),
        "model": model,
        "checkpoint": checkpoint,
        "checkpoint_step": rg.checkpoint_step(checkpoint),
        "layer": layer,
        "layer_depth": layer_depth,
        "feature_kind": feature_kind,
        "feature_dim": int(feature_dim),
        "source_direction_count": int(source_direction_count),
        "variance_component_count": variance_component_count,
        "residual_variance_explained": residual_variance_explained,
        "direction_variance_explained": direction_variance_explained,
        "train_n": int(train_n),
        "test_n": int(test_n),
        "target_direction_capture": target_direction_capture,
        "source_target_direction_cosine": source_target_direction_cosine,
        "is_target_aggregate": is_target_aggregate,
        "n_target_folds": n_target_folds,
    }


def load_grouped_entries(
    representation_index: Dict[Tuple[str, str, str], Path],
    model: str,
    checkpoint: str,
) -> Dict[str, Dict[str, object]]:
    entries = {}
    grouped = rg.grouped_representation_paths(representation_index)
    for (dataset, entry_model, entry_checkpoint), paths in sorted(grouped.items()):
        if entry_model != model or entry_checkpoint != checkpoint:
            continue
        ids, labels, embeddings, source_datasets = rg.load_grouped_representations(paths)
        if len(labels) == 0 or embeddings.size == 0:
            continue
        entries[dataset] = {
            "ids": ids,
            "labels": labels.astype(int),
            "embeddings": embeddings,
            "source_datasets": source_datasets,
            "is_aggregate": dataset == "temporal_concord" and len(source_datasets) > 1,
            "n_folds": float(len(source_datasets)) if dataset == "temporal_concord" and len(source_datasets) > 1 else math.nan,
        }
    return entries


def compute_direction_subspace_probes(
    train_index: Dict[Tuple[str, str, str], Path],
    test_index: Dict[Tuple[str, str, str], Path],
    requested_layers_arg: Optional[Sequence[int]],
    max_subspace_dims: int,
    include_full_probe: bool,
    seed: int,
) -> pd.DataFrame:
    records = []
    keys = sorted(set(train_index).intersection(set(test_index)))
    model_names = rg.sort_names_with_architecture_hint({model for _, model, _ in keys})
    checkpoints = sorted({checkpoint for _, _, checkpoint in keys}, key=rg.checkpoint_sort_key)

    for model in model_names:
        for checkpoint in checkpoints:
            train_entries = load_grouped_entries(train_index, model, checkpoint)
            test_entries = load_grouped_entries(test_index, model, checkpoint)
            datasets = sorted(set(train_entries).intersection(set(test_entries)))
            if len(datasets) == 0:
                continue

            first_dataset = datasets[0]
            num_layers = train_entries[first_dataset]["embeddings"].shape[1]
            layers = rg.selected_layers(num_layers, requested_layers_arg)

            for layer in layers:
                direction_info = {}
                for dataset in datasets:
                    X_train = train_entries[dataset]["embeddings"][:, layer, :]
                    y_train = train_entries[dataset]["labels"]
                    stats, unit_direction, _ = rg.preference_direction_stats(X_train, y_train)
                    if len(unit_direction) == 0:
                        continue
                    direction_info[dataset] = {
                        "unit_direction": unit_direction.astype(np.float64),
                        "direction_norm": stats["direction_norm"],
                    }

                if not direction_info:
                    continue

                for target_dataset in datasets:
                    if target_dataset not in direction_info:
                        continue
                    target_train = train_entries[target_dataset]
                    target_test = test_entries[target_dataset]
                    X_train = target_train["embeddings"][:, layer, :]
                    X_test = target_test["embeddings"][:, layer, :]
                    y_train = target_train["labels"]
                    y_test = target_test["labels"]
                    target_unit = direction_info[target_dataset]["unit_direction"]
                    layer_depth = rg.layer_depth(layer, num_layers)
                    is_aggregate = bool(target_train["is_aggregate"])
                    n_folds = float(target_train["n_folds"])

                    for source_dataset in datasets:
                        source_direction = direction_info.get(source_dataset)
                        if source_direction is None:
                            continue
                        basis = source_direction["unit_direction"].reshape(-1, 1)
                        Xtr, Xte = projection_features(X_train, X_test, basis)
                        metrics = probe_features(Xtr, y_train, Xte, y_test, seed)
                        record = direction_record_base(
                            target_dataset=target_dataset,
                            source_dataset=source_dataset,
                            source_datasets=[source_dataset],
                            model=model,
                            checkpoint=checkpoint,
                            layer=layer,
                            layer_depth=layer_depth,
                            feature_kind="single_direction",
                            feature_dim=1,
                            source_direction_count=1,
                            train_n=len(y_train),
                            test_n=len(y_test),
                            target_direction_capture=direction_capture(basis, target_unit),
                            source_target_direction_cosine=direction_cosine(
                                source_direction["unit_direction"],
                                target_unit,
                            ),
                            is_target_aggregate=is_aggregate,
                            n_target_folds=n_folds,
                        )
                        records.append({**record, **metrics})

                    basis, variance_stats = direction_plus_variance_basis(
                        X_train,
                        target_unit,
                        n_variance_dims=max_subspace_dims,
                    )
                    if basis.size > 0 and basis.shape[1] > 1:
                        Xtr, Xte = projection_features(X_train, X_test, basis)
                        metrics = probe_features(Xtr, y_train, Xte, y_test, seed)
                        record = direction_record_base(
                            target_dataset=target_dataset,
                            source_dataset=target_dataset,
                            source_datasets=[target_dataset],
                            model=model,
                            checkpoint=checkpoint,
                            layer=layer,
                            layer_depth=layer_depth,
                            feature_kind="direction_plus_variance",
                            feature_dim=basis.shape[1],
                            source_direction_count=1,
                            variance_component_count=variance_stats["variance_component_count"],
                            residual_variance_explained=variance_stats["residual_variance_explained"],
                            direction_variance_explained=variance_stats["direction_variance_explained"],
                            train_n=len(y_train),
                            test_n=len(y_test),
                            target_direction_capture=direction_capture(basis, target_unit),
                            source_target_direction_cosine=1.0,
                            is_target_aggregate=is_aggregate,
                            n_target_folds=n_folds,
                        )
                        records.append({**record, **metrics})

                    if include_full_probe:
                        metrics = probe_features(
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            seed,
                        )
                        record = direction_record_base(
                            target_dataset=target_dataset,
                            source_dataset="full_representation",
                            source_datasets=[],
                            model=model,
                            checkpoint=checkpoint,
                            layer=layer,
                            layer_depth=layer_depth,
                            feature_kind="full_representation",
                            feature_dim=X_train.shape[1],
                            source_direction_count=0,
                            train_n=len(y_train),
                            test_n=len(y_test),
                            target_direction_capture=1.0,
                            source_target_direction_cosine=math.nan,
                            is_target_aggregate=is_aggregate,
                            n_target_folds=n_folds,
                        )
                        records.append({**record, **metrics})

    return pd.DataFrame(records)


def plot_transfer_heatmaps(df: pd.DataFrame, plots_dir: Path) -> None:
    heatmap_dir = plots_dir / "transfer_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    single = df[df["feature_kind"] == "single_direction"].copy()
    single = single[np.isclose(single["layer_depth"], 1.0)]
    if single.empty:
        return

    for checkpoint, checkpoint_df in single.groupby("checkpoint", dropna=False):
        matrix = checkpoint_df.pivot_table(
            index="source_dataset",
            columns="target_dataset",
            values="accuracy",
            aggfunc="mean",
        )
        if matrix.empty:
            continue
        datasets = sorted(set(matrix.index).union(set(matrix.columns)))
        matrix = matrix.reindex(index=datasets, columns=datasets)

        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Probe accuracy"},
        )
        ax.set_title(f"Single task-direction probe transfer ({checkpoint})", fontsize=13)
        ax.set_xlabel("Target task")
        ax.set_ylabel("Source direction")
        fig.tight_layout()
        out = heatmap_dir / f"direction_probe_transfer_{rg.checkpoint_plot_label(checkpoint)}.svg"
        fig.savefig(out, dpi=220)
        plt.close(fig)


def plot_variance_augmented_layers(df: pd.DataFrame, plots_dir: Path) -> None:
    augmented_dir = plots_dir / "variance_augmented_layers"
    augmented_dir.mkdir(parents=True, exist_ok=True)
    subset = df[df["feature_kind"].isin(["single_direction", "direction_plus_variance"])].copy()
    subset = subset[subset["source_dataset"] == subset["target_dataset"]]
    if subset.empty:
        return

    for checkpoint, checkpoint_df in subset.groupby("checkpoint", dropna=False):
        dataset_names = sorted(checkpoint_df["target_dataset"].unique().tolist())
        feature_labels = {
            "single_direction": "1D direction",
            "direction_plus_variance": "1D direction + PCA variance",
        }
        color_map = {
            "single_direction": sns.color_palette("tab10", n_colors=2)[0],
            "direction_plus_variance": sns.color_palette("tab10", n_colors=2)[1],
        }
        fig, axes = plt.subplots(nrows=1, ncols=len(dataset_names), figsize=(5.4 * len(dataset_names), 4.2), sharey=True)
        axes = [axes] if len(dataset_names) == 1 else list(axes)

        for ax, dataset in zip(axes, dataset_names):
            data_for_dataset = checkpoint_df[checkpoint_df["target_dataset"] == dataset]
            for feature_kind, label in feature_labels.items():
                data = (
                    data_for_dataset[data_for_dataset["feature_kind"] == feature_kind]
                    .groupby("layer_depth", as_index=False)
                    .agg(accuracy=("accuracy", "mean"), accuracy_std=("accuracy", "std"))
                    .sort_values("layer_depth")
                )
                if data.empty:
                    continue
                ax.plot(
                    data["layer_depth"],
                    data["accuracy"],
                    color=color_map[feature_kind],
                    marker="o",
                    linewidth=1.9,
                    markersize=3.5,
                    alpha=0.9,
                    label=label,
                )
                std = data["accuracy_std"].fillna(0.0)
                ax.fill_between(
                    data["layer_depth"],
                    data["accuracy"] - std,
                    data["accuracy"] + std,
                    color=color_map[feature_kind],
                    alpha=0.12,
                    linewidth=0,
                )
            rg.style_plain_ax(ax, "Relative depth", "Probe accuracy")
            ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1.0, alpha=0.8)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(0.0, 1.02)
            ax.set_title(dataset, fontsize=13)

        handles = [
            plt.Line2D([0], [0], color=color_map[kind], marker="o", linewidth=1.9, markersize=3.5, label=label)
            for kind, label in feature_labels.items()
        ]
        fig.suptitle(f"Direction plus residual PCA probe ({checkpoint})", fontsize=16)
        fig.legend(handles, [handle.get_label() for handle in handles], loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.08, 1, 0.94])
        out = augmented_dir / f"direction_probe_variance_augmented_layers_{rg.checkpoint_plot_label(checkpoint)}.svg"
        fig.savefig(out, dpi=220)
        plt.close(fig)


def plot_layer_scores(df: pd.DataFrame, plots_dir: Path) -> None:
    layer_dir = plots_dir / "layer_scores"
    layer_dir.mkdir(parents=True, exist_ok=True)
    own = df[
        (df["feature_kind"] == "single_direction")
        & (df["source_dataset"] == df["target_dataset"])
    ].copy()
    if own.empty:
        return

    for checkpoint, checkpoint_df in own.groupby("checkpoint", dropna=False):
        dataset_names = sorted(checkpoint_df["target_dataset"].unique().tolist())
        models = rg.sort_names_with_architecture_hint(checkpoint_df["model"].unique().tolist())
        palette = sns.color_palette("tab10", n_colors=len(models))
        color_map = dict(zip(models, palette))
        fig, axes = plt.subplots(nrows=1, ncols=len(dataset_names), figsize=(5.4 * len(dataset_names), 4.2), sharey=True)
        axes = [axes] if len(dataset_names) == 1 else list(axes)

        for ax, dataset in zip(axes, dataset_names):
            subset = checkpoint_df[checkpoint_df["target_dataset"] == dataset]
            for model in models:
                model_data = subset[subset["model"] == model].sort_values("layer_depth")
                if model_data.empty:
                    continue
                ax.plot(
                    model_data["layer_depth"],
                    model_data["accuracy"],
                    color=color_map[model],
                    marker="o",
                    linewidth=1.9,
                    markersize=3.5,
                    alpha=0.9,
                    label=rg.short_model_name(model),
                )
            rg.style_plain_ax(ax, "Relative depth", "Probe accuracy")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(0.0, 1.02)
            ax.set_title(dataset, fontsize=13)

        handles = [
            plt.Line2D([0], [0], color=color_map[model], marker="o", linewidth=1.9, markersize=3.5, label=rg.short_model_name(model))
            for model in models
        ]
        fig.suptitle(f"Own direction 1D probe ({checkpoint})", fontsize=16)
        fig.legend(handles, [handle.get_label() for handle in handles], loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.08, 1, 0.94])
        out = layer_dir / f"direction_probe_own_direction_layers_{rg.checkpoint_plot_label(checkpoint)}.svg"
        fig.savefig(out, dpi=220)
        plt.close(fig)


def plot_results(df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_transfer_heatmaps(df, plots_dir)
    plot_layer_scores(df, plots_dir)
    plot_variance_augmented_layers(df, plots_dir)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_specs = discover_split_datasets(Path(args.train_dir), "train", args.datasets)
    test_specs = discover_split_datasets(Path(args.test_dir), "test", args.datasets)
    train_data = load_split_data(train_specs, args.max_samples, args.seed)
    test_data = load_split_data(test_specs, args.max_samples, args.seed)

    model_names = rg.discover_model_names(Path(args.models_dir), args.model_names)
    checkpoint_labels = rg.requested_checkpoint_labels(args)
    if args.checkpoints is None:
        args.include_final = True

    train_index = rg.ensure_representation_caches(
        args=args,
        model_names=model_names,
        checkpoint_labels=checkpoint_labels,
        dataset_specs=train_specs,
        datasets=train_data,
    )
    test_index = rg.ensure_representation_caches(
        args=args,
        model_names=model_names,
        checkpoint_labels=checkpoint_labels,
        dataset_specs=test_specs,
        datasets=test_data,
    )

    df = compute_direction_subspace_probes(
        train_index=train_index,
        test_index=test_index,
        requested_layers_arg=args.layers,
        max_subspace_dims=args.max_subspace_dims,
        include_full_probe=args.include_full_probe,
        seed=args.seed,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "direction_subspace_probe.csv"
    rg.write_csv(df, output_path, RESULT_COLUMNS)

    if args.plot:
        plot_results(df, Path(args.plots_dir))


if __name__ == "__main__":
    main()
