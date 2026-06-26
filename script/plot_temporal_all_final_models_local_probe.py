from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
import re


MODEL_RE = re.compile(r"train__temporal_concord_fold_0__(.+)__final__all\.npz$")
NEIGHBOR_METRIC = "cosine"


def warn(message):
    print(f"[temporal_all_final_models] {message}", flush=True)


def model_sort_key(model):
    match = re.search(r"gpt_(\d+)l_(\d+)h_(\d+)d", model)
    if not match:
        return (999, 999, 999, model)
    layers, heads, width = (int(group) for group in match.groups())
    return (layers, heads, width, model)


def discover_complete_models(representations_dir):
    models = []
    for file_name in os.listdir(representations_dir):
        match = MODEL_RE.match(file_name)
        if not match:
            continue

        model = match.group(1)
        complete = True
        for fold in range(5):
            for split in ["train", "test"]:
                expected = os.path.join(
                    representations_dir,
                    f"{split}__temporal_concord_fold_{fold}__{model}__final__all.npz",
                )
                if not os.path.exists(expected):
                    complete = False
        if complete:
            models.append(model)

    return sorted(models, key=model_sort_key)


def load_embeddings(representations_dir, split, fold, model):
    file_name = f"{split}__temporal_concord_fold_{fold}__{model}__final__all.npz"
    path = os.path.join(representations_dir, file_name)
    with np.load(path, allow_pickle=False) as data:
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=int)
    return embeddings, labels


def local_centroid_predictions(X_train, y_train, X_test, neighbor_indices, fallback_label):
    pred = np.empty(len(X_test), dtype=int)
    for row, local_idx in enumerate(neighbor_indices):
        labels = y_train[local_idx]
        has0 = np.any(labels == 0)
        has1 = np.any(labels == 1)
        if not (has0 and has1):
            pred[row] = int(labels[0]) if len(labels) else fallback_label
            continue

        local_X = X_train[local_idx]
        mean0 = local_X[labels == 0].mean(axis=0)
        mean1 = local_X[labels == 1].mean(axis=0)
        direction = mean1 - mean0
        midpoint = 0.5 * (mean0 + mean1)
        pred[row] = int((X_test[row] - midpoint) @ direction > 0)

    return pred


def macro_f1(y, pred):
    return float(f1_score(y, pred, average="macro", zero_division=0))


def majority_label(labels):
    counts = np.bincount(labels.astype(int), minlength=2)
    return int(counts[1] > counts[0])


def evaluate_fold_layer(train_embeddings, train_labels, test_embeddings, test_labels, layer, k):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_embeddings[:, layer, :]).astype(np.float32)
    X_test = scaler.transform(test_embeddings[:, layer, :]).astype(np.float32)

    effective_k = min(k, len(train_labels))
    nn = NearestNeighbors(n_neighbors=effective_k, algorithm="brute", metric=NEIGHBOR_METRIC, n_jobs=-1)
    nn.fit(X_train)
    neighbor_indices = nn.kneighbors(X_test, return_distance=False)
    local_pred = local_centroid_predictions(
        X_train,
        train_labels,
        X_test,
        neighbor_indices,
        majority_label(train_labels),
    )

    global_probe = RidgeClassifier(class_weight="balanced", random_state=42)
    global_probe.fit(X_train, train_labels)
    global_pred = global_probe.predict(X_test)

    return {
        "local_centroid_macro_f1": macro_f1(test_labels, local_pred),
        "global_ridge_macro_f1": macro_f1(test_labels, global_pred),
    }


def compute_results(representations_dir, models, k):
    records = []
    for model in models:
        warn(f"Evaluating {model}")
        fold_cache = []
        num_layers = None

        for fold in range(5):
            train_embeddings, train_labels = load_embeddings(representations_dir, "train", fold, model)
            test_embeddings, test_labels = load_embeddings(representations_dir, "test", fold, model)
            if num_layers is None:
                num_layers = int(train_embeddings.shape[1])
            elif num_layers != int(train_embeddings.shape[1]):
                raise ValueError(f"Layer mismatch for {model}")
            fold_cache.append((fold, train_embeddings, train_labels, test_embeddings, test_labels))

        for layer in range(num_layers):
            for fold, train_embeddings, train_labels, test_embeddings, test_labels in fold_cache:
                scores = evaluate_fold_layer(
                    train_embeddings,
                    train_labels,
                    test_embeddings,
                    test_labels,
                    layer,
                    k,
                )
                relative_depth = layer / (num_layers - 1) if num_layers > 1 else 0.0
                local_label = f"Local centroid, train k={k}"

                for method, value in [
                    (local_label, scores["local_centroid_macro_f1"]),
                    ("Global Ridge probe", scores["global_ridge_macro_f1"]),
                ]:
                    records.append({
                        "dataset": "temporal_concord",
                        "model": model,
                        "method_label": method,
                        "fold": fold,
                        "layer": layer,
                        "num_layers": num_layers,
                        "relative_depth": relative_depth,
                        "neighbor_metric": NEIGHBOR_METRIC if method.startswith("Local") else "none",
                        "requested_k": k if method.startswith("Local") else np.nan,
                        "f1_macro": value,
                    })

    return pd.DataFrame(records)


def aggregate_results(fold_results):
    group_cols = [
        "dataset",
        "model",
        "method_label",
        "layer",
        "num_layers",
        "relative_depth",
        "neighbor_metric",
        "requested_k",
    ]
    return (
        fold_results.groupby(group_cols, dropna=False, as_index=False)
        .agg(f1_macro=("f1_macro", "mean"), f1_macro_std=("f1_macro", "std"), n_folds=("fold", "nunique"))
        .sort_values(["method_label", "model", "layer"])
    )


def style_ax(ax, xlabel, ylabel):
    ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def get_colors(models):
    cmap = plt.get_cmap("tab10")
    return {model: cmap(idx % 10) for idx, model in enumerate(models)}


def plot_results(summary, output_dir, stem, k):
    models = sorted(summary["model"].unique().tolist(), key=model_sort_key)
    color_map = get_colors(models)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13.2, 4.8), sharey=True, constrained_layout=True)
    for ax, method in zip(axes, [f"Local centroid, train k={k}", "Global Ridge probe"]):
        method_df = summary[summary["method_label"] == method]
        for model in models:
            model_df = method_df[method_df["model"] == model].sort_values("relative_depth")
            if model_df.empty:
                continue
            ax.plot(
                model_df["relative_depth"],
                model_df["f1_macro"],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                color=color_map[model],
                label=model,
            )

        style_ax(ax, "Relative Depth", "Macro F1")
        ax.set_ylim(0.30, 0.95)
        ax.set_title(method, fontsize=12)

    handles = [
        plt.Line2D([0], [0], color=color_map[model], marker="o", linewidth=1.8, markersize=3.5, label=model)
        for model in models
    ]
    fig.suptitle("Temporal Concord", fontsize=14)
    fig.legend(handles, models, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.savefig(os.path.join(output_dir, f"{stem}.svg"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{stem}.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_final_layer(summary, output_dir, stem, k):
    final = summary[summary["layer"] == (summary["num_layers"] - 1)].copy()
    models = sorted(final["model"].unique().tolist(), key=model_sort_key)
    local_label = f"Local centroid, train k={k}"
    methods = ["Global Ridge probe", local_label]
    colors = {
        "Global Ridge probe": "#72b6a1",
        local_label: "#e99675",
    }

    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    x = np.arange(len(models), dtype=float)
    width = 0.36
    offsets = {
        "Global Ridge probe": -width / 2,
        local_label: width / 2,
    }

    for method in methods:
        method_df = final[final["method_label"] == method].set_index("model")
        values = [float(method_df.loc[model, "f1_macro"]) if model in method_df.index else np.nan for model in models]
        ax.bar(x + offsets[method], values, width=width, color=colors[method], label=method)

    style_ax(ax, "", "Final-Layer Macro F1")
    ax.set_title("Temporal Concord", fontsize=13)
    ax.set_ylim(0.30, 0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(os.path.join(output_dir, f"{stem}_final_layer.svg"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{stem}_final_layer.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser("Temporal all-final-model local/global macro-F1 comparison.")
    parser.add_argument("--representations_dir", type=str, default="data/representations_geometry")
    parser.add_argument("--output_dir", type=str, default="plots/local_probing")
    parser.add_argument("--results_dir", type=str, default="data/results/train_reference_local_probe")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    models = args.models or discover_complete_models(args.representations_dir)
    if not models:
        raise ValueError(f"No complete final temporal models found in {args.representations_dir}")

    fold_results = compute_results(args.representations_dir, models, args.k)
    summary = aggregate_results(fold_results)

    stem = f"temporal_all_final_models_local_vs_global_macro_f1_k{args.k}_{NEIGHBOR_METRIC}"
    fold_path = os.path.join(args.results_dir, f"{stem}_folds.csv")
    summary_path = os.path.join(args.results_dir, f"{stem}_summary.csv")
    fold_results.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)

    plot_results(summary, args.output_dir, stem, args.k)
    plot_final_layer(summary, args.output_dir, stem, args.k)

    warn(f"Saved fold results to {fold_path}")
    warn(f"Saved summary to {summary_path}")
    warn(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
