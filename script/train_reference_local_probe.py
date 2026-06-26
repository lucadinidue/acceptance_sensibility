from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import argparse
import math
import os
import re


NEIGHBOR_METRIC = "cosine"


def warn(message):
    print(f"[train_reference_local_probe] {message}", flush=True)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "value"


def get_layer(layer, num_layers):
    actual_layer = layer
    if actual_layer < 0:
        actual_layer = num_layers + actual_layer
    if actual_layer < 0 or actual_layer >= num_layers:
        raise ValueError(f"Layer {layer} resolves to {actual_layer}, but there are {num_layers} layers.")
    return int(actual_layer)


def get_dataset_parts(dataset):
    if dataset == "temporal_concord":
        return [(f"temporal_concord_fold_{fold}", fold) for fold in range(5)]
    return [(dataset, None)]


def load_representations(representations_dir, split, part_name, model_name, layer):
    file_name = f"{split}__{part_name}__{model_name}__final__all.npz"
    path = os.path.join(representations_dir, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        ids = np.asarray(data["ids"], dtype=str)
        labels = np.asarray(data["labels"], dtype=int)
        sentences = np.asarray(data["sentences"], dtype=str)
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)

    actual_layer = get_layer(layer, embeddings.shape[1])
    return {
        "ids": ids,
        "labels": labels,
        "sentences": sentences,
        "embeddings": embeddings[:, actual_layer, :].astype(np.float32, copy=False),
        "layer": actual_layer,
        "num_layers": int(embeddings.shape[1]),
    }


def get_f1_scores(y, pred):
    return {
        "f1_class0": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "f1_class1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def get_auc(y, score):
    try:
        return float(roc_auc_score(y, score))
    except ValueError:
        return math.nan


def majority_label(labels):
    counts = np.bincount(labels.astype(int), minlength=2)
    return int(counts[1] > counts[0])


def local_centroid_predictions(X_train, y_train, X_test, neighbor_indices, fallback_label):
    pred = np.empty(len(X_test), dtype=int)
    score = np.empty(len(X_test), dtype=np.float64)
    one_class = np.zeros(len(X_test), dtype=bool)

    for row, local_idx in enumerate(neighbor_indices):
        labels = y_train[local_idx]
        has0 = np.any(labels == 0)
        has1 = np.any(labels == 1)
        if not (has0 and has1):
            one_class[row] = True
            pred[row] = int(labels[0]) if len(labels) else fallback_label
            score[row] = float(pred[row])
            continue

        local_X = X_train[local_idx]
        mean0 = local_X[labels == 0].mean(axis=0)
        mean1 = local_X[labels == 1].mean(axis=0)
        direction = mean1 - mean0
        midpoint = 0.5 * (mean0 + mean1)
        score[row] = float((X_test[row] - midpoint) @ direction)
        pred[row] = int(score[row] > 0)

    return pred, score, one_class


def make_summary_row(dataset, model_name, layer, method, requested_k, effective_k,
                     fold, y, pred, score, one_class_rate=math.nan):
    row = {
        "dataset": dataset,
        "model": model_name,
        "layer": layer,
        "method": method,
        "neighbor_metric": NEIGHBOR_METRIC if method == "local_centroid" else "none",
        "requested_k": requested_k,
        "effective_k": effective_k,
        "fold": fold,
        "n": int(len(y)),
        "acc": float(accuracy_score(y, pred)),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "auc_class1_high": get_auc(y, score),
        "one_class_neighborhood_rate": one_class_rate,
    }
    row.update(get_f1_scores(y, pred))
    return row


def evaluate_part(dataset, part_name, fold, args):
    train = load_representations(args.representations_dir, "train", part_name, args.model_name, args.layer)
    test = load_representations(args.representations_dir, "test", part_name, args.model_name, args.layer)

    if train["layer"] != test["layer"]:
        raise ValueError(f"Layer mismatch for {part_name}: train {train['layer']}, test {test['layer']}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train["embeddings"]).astype(np.float32)
    X_test = scaler.transform(test["embeddings"]).astype(np.float32)

    global_probe = RidgeClassifier(class_weight="balanced", random_state=42)
    global_probe.fit(X_train, train["labels"])
    global_pred = global_probe.predict(X_test)
    global_score = global_probe.decision_function(X_test)

    fold_label = fold if fold is not None else "all"
    summary_rows = [
        make_summary_row(dataset, args.model_name, train["layer"], "global_ridge",
                         math.nan, math.nan, fold_label, test["labels"], global_pred, global_score)
    ]
    sentence_frames = []

    max_k = min(max(args.k_values), len(train["labels"]))
    nn = NearestNeighbors(n_neighbors=max_k, algorithm="brute", metric=NEIGHBOR_METRIC, n_jobs=args.n_jobs)
    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_test, return_distance=True)
    fallback = majority_label(train["labels"])

    for requested_k in args.k_values:
        effective_k = min(requested_k, indices.shape[1])
        local_idx = indices[:, :effective_k]
        local_pred, local_score, one_class = local_centroid_predictions(
            X_train, train["labels"], X_test, local_idx, fallback
        )
        summary_rows.append(
            make_summary_row(dataset, args.model_name, train["layer"], "local_centroid",
                             float(requested_k), float(effective_k), fold_label,
                             test["labels"], local_pred, local_score, float(np.mean(one_class)))
        )

        if args.save_sentences:
            sentence_frames.append(pd.DataFrame({
                "dataset": dataset,
                "fold": fold,
                "source_part": part_name,
                "model": args.model_name,
                "layer": train["layer"],
                "neighbor_metric": NEIGHBOR_METRIC,
                "requested_k": requested_k,
                "effective_k": effective_k,
                "id": test["ids"],
                "sentence": test["sentences"],
                "acceptability": test["labels"],
                "global_ridge_pred": global_pred,
                "global_ridge_score": global_score,
                "local_centroid_pred": local_pred,
                "local_centroid_score": local_score,
                "local_centroid_one_class_neighborhood": one_class,
                "nearest_train_distance": distances[:, 0],
                "kth_train_distance": distances[:, effective_k - 1],
            }))

    sentence_df = pd.concat(sentence_frames, ignore_index=True) if sentence_frames else pd.DataFrame()
    return pd.DataFrame(summary_rows), sentence_df, train["layer"]


def aggregate_summary(fold_summary):
    group_cols = ["dataset", "model", "layer", "method", "neighbor_metric", "requested_k", "effective_k"]
    metric_cols = [
        "acc",
        "bal_acc",
        "f1_class0",
        "f1_class1",
        "f1_macro",
        "auc_class1_high",
        "one_class_neighborhood_rate",
    ]

    rows = []
    for key, group in fold_summary.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["fold"] = "mean_5folds" if row["dataset"] == "temporal_concord" else "all"
        row["n"] = int(group["n"].sum())
        for col in metric_cols:
            values = group[col].to_numpy(dtype=float)
            row[col] = float(np.nanmean(values)) if np.isfinite(values).any() else math.nan
            row[f"{col}_std"] = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser("Train-reference local centroid probe.")
    parser.add_argument("--representations_dir", type=str, default="data/representations_geometry")
    parser.add_argument("--output_dir", type=str, default="data/results/train_reference_local_probe")
    parser.add_argument("--model_name", type=str, default="gpt_8l_8h_512d_42s")
    parser.add_argument("--datasets", nargs="+", default=["temporal_concord"])
    parser.add_argument("--k_values", nargs="+", type=int, default=[100])
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--save_sentences", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_summary = []
    all_sentences = []
    actual_layers = []
    for dataset in args.datasets:
        for part_name, fold in get_dataset_parts(dataset):
            warn(f"Evaluating {dataset} fold={fold} layer={args.layer}")
            summary, sentences, actual_layer = evaluate_part(dataset, part_name, fold, args)
            all_summary.append(summary)
            if len(sentences):
                all_sentences.append(sentences)
            actual_layers.append(actual_layer)

    fold_summary = pd.concat(all_summary, ignore_index=True)
    summary = aggregate_summary(fold_summary)

    unique_layers = sorted(set(actual_layers))
    layer_tag = str(unique_layers[0]) if len(unique_layers) == 1 else "_".join(str(layer) for layer in unique_layers)
    k_tag = "_".join(str(k) for k in args.k_values)
    dataset_tag = "_".join(safe_name(dataset) for dataset in args.datasets)
    stem = (
        f"train_reference_local_probe_{dataset_tag}_{safe_name(args.model_name)}_"
        f"layer{layer_tag}_k{k_tag}_{NEIGHBOR_METRIC}"
    )

    fold_path = os.path.join(args.output_dir, f"{stem}_fold_summary.csv")
    summary_path = os.path.join(args.output_dir, f"{stem}_summary.csv")
    fold_summary.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    warn(f"Saved fold summary to {fold_path}")
    warn(f"Saved summary to {summary_path}")

    if all_sentences:
        sentence_path = os.path.join(args.output_dir, f"{stem}_sentences.csv")
        pd.concat(all_sentences, ignore_index=True).to_csv(sentence_path, index=False)
        warn(f"Saved sentence predictions to {sentence_path}")


if __name__ == "__main__":
    main()
