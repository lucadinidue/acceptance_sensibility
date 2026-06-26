"""Label diagnostics for local geometry in acceptance datasets.

This script reuses the cached sentence representations used by the within-
dataset ESS analysis. For each dataset it builds neighborhoods from all other
sentences in that dataset, ignoring labels during neighbor search, then checks
whether labels are locally clustered. It also runs a simple cross-validated
linear probe on the same embeddings.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from within_dataset_pointwise_ess import load_datasets, safe_name, within_dataset_neighbors


def warn(message: str) -> None:
    print(f"[local_geometry] {message}", flush=True)


def class_counts(y: np.ndarray) -> np.ndarray:
    return np.bincount(y.astype(int), minlength=2)


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, score))
    except ValueError:
        return math.nan


def same_label_baseline(y: np.ndarray) -> np.ndarray:
    counts = class_counts(y)
    n = len(y)
    if n <= 1:
        return np.full(n, math.nan)
    return np.asarray([(counts[label] - 1) / (n - 1) for label in y], dtype=np.float64)


def class_accuracy(y: np.ndarray, pred: np.ndarray, label: int) -> float:
    mask = y == label
    if not np.any(mask):
        return math.nan
    return float(np.mean(pred[mask] == y[mask]))


def f1_metrics(y: np.ndarray, pred: np.ndarray, prefix: str) -> dict[str, float]:
    stem = f"{prefix}_" if prefix else ""
    return {
        f"{stem}f1_class0": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        f"{stem}f1_class1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        f"{stem}f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def split_iterator(
    dataset: str,
    rows: pd.DataFrame,
    y: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    if dataset == "temporal_concord" and rows["fold"].notna().any():
        groups = rows["fold"].to_numpy()
        return LeaveOneGroupOut().split(np.zeros(len(y)), y, groups)
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    ).split(np.zeros(len(y)), y)


def global_majority_label(y: np.ndarray) -> int:
    counts = class_counts(y)
    return int(counts[1] > counts[0])


def neighbor_diagnostics(
    rows: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    k_values: list[int],
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sentence_frames = []
    summary_rows = []

    for dataset, dataset_rows in rows.groupby("dataset", sort=True):
        dataset_indices = dataset_rows.index.to_numpy()
        dataset_embeddings = embeddings[dataset_indices]
        y = dataset_rows["acceptability"].to_numpy(dtype=int)
        counts = class_counts(y)
        majority_label = global_majority_label(y)
        majority_acc = float(np.max(counts) / len(y))
        random_same = same_label_baseline(y)

        warn(f"Neighbor diagnostics for {dataset}: {len(dataset_rows)} sentences")
        for requested_k in k_values:
            neighbor_indices, distances, effective_k = within_dataset_neighbors(
                dataset_embeddings,
                k=requested_k,
                n_jobs=n_jobs,
            )
            neighbor_labels = y[neighbor_indices]
            class1_neighbor_fraction = np.mean(neighbor_labels, axis=1)
            same_label_fraction = np.mean(neighbor_labels == y[:, None], axis=1)
            tie_mask = class1_neighbor_fraction == 0.5
            pred = np.where(
                class1_neighbor_fraction > 0.5,
                1,
                np.where(class1_neighbor_fraction < 0.5, 0, majority_label),
            ).astype(int)

            per_sentence = dataset_rows.copy()
            per_sentence["reference_mode"] = "within_dataset_excluding_self"
            per_sentence["requested_k"] = requested_k
            per_sentence["effective_k"] = effective_k
            per_sentence["reference_count"] = len(dataset_rows) - 1
            per_sentence["same_label_fraction"] = same_label_fraction
            per_sentence["expected_same_label_fraction"] = random_same
            per_sentence["same_label_excess"] = same_label_fraction - random_same
            per_sentence["class1_neighbor_fraction"] = class1_neighbor_fraction
            per_sentence["knn_pred_acceptability"] = pred
            per_sentence["knn_vote_tie"] = tie_mask
            per_sentence["nearest_neighbor_distance"] = distances[:, 0]
            per_sentence["kth_neighbor_distance"] = distances[:, effective_k - 1]
            sentence_frames.append(per_sentence)

            for label in [0, 1]:
                label_mask = y == label
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "requested_k": requested_k,
                        "effective_k": effective_k,
                        "scope": f"class_{label}",
                        "n": int(np.sum(label_mask)),
                        "same_label_fraction": float(np.mean(same_label_fraction[label_mask])),
                        "expected_same_label_fraction": float(np.mean(random_same[label_mask])),
                        "same_label_excess": float(
                            np.mean(same_label_fraction[label_mask] - random_same[label_mask])
                        ),
                        "class1_neighbor_fraction": float(
                            np.mean(class1_neighbor_fraction[label_mask])
                        ),
                        "knn_acc": class_accuracy(y, pred, label),
                        "knn_bal_acc": math.nan,
                        "knn_f1_class0": math.nan,
                        "knn_f1_class1": math.nan,
                        "knn_f1_macro": math.nan,
                        "knn_auc_class1_high": math.nan,
                        "knn_auc_best_direction": math.nan,
                        "majority_acc": majority_acc,
                        "tie_rate": float(np.mean(tie_mask[label_mask])),
                    }
                )

            auc = safe_auc(y, class1_neighbor_fraction)
            knn_f1 = f1_metrics(y, pred, "knn")
            summary_rows.append(
                {
                    "dataset": dataset,
                    "requested_k": requested_k,
                    "effective_k": effective_k,
                    "scope": "all",
                    "n": int(len(y)),
                    "same_label_fraction": float(np.mean(same_label_fraction)),
                    "expected_same_label_fraction": float(np.mean(random_same)),
                    "same_label_excess": float(np.mean(same_label_fraction - random_same)),
                    "class1_neighbor_fraction": float(np.mean(class1_neighbor_fraction)),
                    "knn_acc": float(accuracy_score(y, pred)),
                    "knn_bal_acc": float(balanced_accuracy_score(y, pred)),
                    **knn_f1,
                    "knn_auc_class1_high": auc,
                    "knn_auc_best_direction": float(max(auc, 1 - auc)) if np.isfinite(auc) else math.nan,
                    "majority_acc": majority_acc,
                    "tie_rate": float(np.mean(tie_mask)),
                }
            )

    return pd.concat(sentence_frames, ignore_index=True), pd.DataFrame(summary_rows)


def linear_probe_diagnostics(
    rows: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
    c: float,
    max_iter: int,
    class_weight: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    fold_rows = []

    for dataset, dataset_rows in rows.groupby("dataset", sort=True):
        dataset_indices = dataset_rows.index.to_numpy()
        x = embeddings[dataset_indices].astype(np.float32, copy=False)
        y = dataset_rows["acceptability"].to_numpy(dtype=int)
        counts = class_counts(y)
        majority_acc = float(np.max(counts) / len(y))
        oof_pred = np.empty(len(y), dtype=int)
        oof_score = np.empty(len(y), dtype=np.float64)

        warn(f"Linear probe for {dataset}: {len(dataset_rows)} sentences")
        splits = split_iterator(
            dataset,
            dataset_rows,
            y,
            n_splits=n_splits,
            random_state=random_state,
        )
        for split_id, (train_idx, test_idx) in enumerate(splits):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    random_state=random_state,
                    solver="liblinear",
                ),
            )
            clf.fit(x[train_idx], y[train_idx])
            pred = clf.predict(x[test_idx])
            score = clf.predict_proba(x[test_idx])[:, 1]
            oof_pred[test_idx] = pred
            oof_score[test_idx] = score

            fold_rows.append(
                {
                    "dataset": dataset,
                    "split_id": split_id,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "test_fold": (
                        str(dataset_rows.iloc[test_idx]["fold"].dropna().unique().tolist())
                        if dataset == "temporal_concord"
                        else "stratified"
                    ),
                    "acc": float(accuracy_score(y[test_idx], pred)),
                    "bal_acc": float(balanced_accuracy_score(y[test_idx], pred)),
                    **f1_metrics(y[test_idx], pred, ""),
                    "auc_class1_high": safe_auc(y[test_idx], score),
                }
            )

        auc = safe_auc(y, oof_score)
        cv_f1 = f1_metrics(y, oof_pred, "cv")
        summary_rows.append(
            {
                "dataset": dataset,
                "n": int(len(y)),
                "class0_n": int(counts[0]),
                "class1_n": int(counts[1]),
                "majority_acc": majority_acc,
                "cv_acc": float(accuracy_score(y, oof_pred)),
                "cv_bal_acc": float(balanced_accuracy_score(y, oof_pred)),
                "cv_class0_acc": class_accuracy(y, oof_pred, 0),
                "cv_class1_acc": class_accuracy(y, oof_pred, 1),
                **cv_f1,
                "cv_auc_class1_high": auc,
                "cv_auc_best_direction": float(max(auc, 1 - auc)) if np.isfinite(auc) else math.nan,
                "class_weight": class_weight or "none",
                "C": c,
                "max_iter": max_iter,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Local geometry label diagnostics.")
    parser.add_argument("--eval_dir", type=str, default="data/acceptance_datasets/test")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--geometry_representations_dir", type=str, default="data/representations_geometry")
    parser.add_argument("--training_reference_output_dir", type=str, default="data/results/pointwise_ess")
    parser.add_argument("--output_dir", type=str, default="data/results/local_geometry_label_diagnostics")
    parser.add_argument("--model_name", type=str, default="gpt_8l_8h_512d_42s")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--k_values", nargs="+", type=int, default=[100, 200, 1000])
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--class_weight", choices=["balanced", "none"], default="balanced")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_weight = None if args.class_weight == "none" else args.class_weight
    rows, embeddings, actual_layer, _ = load_datasets(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentence_df, neighbor_summary_df = neighbor_diagnostics(
        rows,
        embeddings,
        k_values=args.k_values,
        n_jobs=args.n_jobs,
    )
    probe_summary_df, probe_fold_df = linear_probe_diagnostics(
        rows,
        embeddings,
        n_splits=args.cv_splits,
        random_state=args.random_state,
        c=args.C,
        max_iter=args.max_iter,
        class_weight=class_weight,
    )

    k_tag = "_".join(str(k) for k in args.k_values)
    stem = f"local_geometry_label_diagnostics_{safe_name(args.model_name)}_layer{actual_layer}_k{k_tag}"
    sentence_path = output_dir / f"{stem}_neighbor_sentences.csv"
    neighbor_summary_path = output_dir / f"{stem}_neighbor_summary.csv"
    probe_summary_path = output_dir / f"{stem}_linear_probe_summary.csv"
    probe_fold_path = output_dir / f"{stem}_linear_probe_folds.csv"

    sentence_df.to_csv(sentence_path, index=False)
    neighbor_summary_df.to_csv(neighbor_summary_path, index=False)
    probe_summary_df.to_csv(probe_summary_path, index=False)
    probe_fold_df.to_csv(probe_fold_path, index=False)

    warn(f"Saved sentence neighbor diagnostics to {sentence_path}")
    warn(f"Saved neighbor summary to {neighbor_summary_path}")
    warn(f"Saved linear probe summary to {probe_summary_path}")
    warn(f"Saved linear probe folds to {probe_fold_path}")


if __name__ == "__main__":
    main()
