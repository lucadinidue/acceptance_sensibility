from utils import load_acceptability_labels, get_last_token_embeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import torch
import os

sns.set_theme(style="whitegrid", font_scale=0.9)


def get_sorted_checkpoints(model_dir):
    checkpoints = []
    for dir_content in os.listdir(model_dir):
        if "checkpoint-" in dir_content:
            checkpoint_num = int(dir_content.split("-")[1])
            checkpoints.append((checkpoint_num, os.path.join(model_dir, dir_content)))
    return sorted(checkpoints, key=lambda x: x[0])


def load_split(data_dir, file_name):
    path = os.path.join(data_dir, file_name)
    df = pd.read_json(path, lines=True)
    labels = load_acceptability_labels(path, df["id"])
    return df["sentence"].tolist(), labels, df


def get_test_categories(dataset_name, test_df):
    if dataset_name.startswith("blimp"):
        category_values = [m["micro_phenomenon"] for m in test_df["metadata"]]
    else:
        category_values = [str(m) for m in test_df["metadata"]]
    return category_values



def probe_all_layers(train_embeddings, train_labels, test_embeddings, test_labels, test_categories):
    num_layers = train_embeddings.shape[1]
    y_train = np.asarray(train_labels).astype(int)
    y_test = np.asarray(test_labels).astype(int)
    test_categories = np.asarray(test_categories)

    results = []
    for layer in range(num_layers):
        X_train = train_embeddings[:, layer, :]
        X_test = test_embeddings[:, layer, :]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        probe = RidgeClassifier(class_weight="balanced", random_state=42)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)

        results.append({
            "layer": layer,
            "category": "all",
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        })

        for value in pd.unique(test_categories):
            mask = test_categories == value
            results.append({
                "layer": layer,
                "category": value,
                "accuracy": accuracy_score(y_test[mask], y_pred[mask]),
                "f1": f1_score(y_test[mask], y_pred[mask], zero_division=0),
            })

    return results

def plot_checkpoint_scores(df, model_name, output_path):
    dataset_names = sorted(df["dataset"].unique().tolist())
    n_datasets = len(dataset_names)
    num_layers = df["layer"].nunique()

    cmap = plt.get_cmap("viridis")
    norm = Normalize(0, num_layers - 1)

    fig, axes = plt.subplots(nrows=1, ncols=n_datasets, figsize=(5.4 * n_datasets, 4.2))

    if n_datasets == 1:
        axes = [axes]

    for ax, dataset_name in zip(axes, dataset_names):
        subset = df[df["dataset"] == dataset_name]

        for layer in sorted(subset["layer"].unique()):
            layer_data = subset[subset["layer"] == layer].sort_values("checkpoint")
            color = cmap(layer / (num_layers - 1))
            ax.plot(layer_data["checkpoint"], layer_data["accuracy"], color=color,
                    marker="o", linewidth=1.9, markersize=3.5, alpha=0.9)

            if (layer_data["accuracy_std"] > 0).any():
                ax.fill_between(layer_data["checkpoint"],
                                 layer_data["accuracy"] - layer_data["accuracy_std"],
                                 layer_data["accuracy"] + layer_data["accuracy_std"],
                                 color=color, alpha=0.15)

        
        ax.set_title(dataset_name, fontsize=13)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Probing accuracy")
        ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Layer (0 = input → final)")

    fig.suptitle(f"Linear Probing across checkpoints — {model_name}", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Linear probing across pretraining checkpoints for a single model.")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to analyze.")
    parser.add_argument("--models_dir", type=str, default="models/pretrained", help="The directory containing the model folders.")
    parser.add_argument("--train_dir", type=str, default="data/acceptance_datasets/train", help="Path of the directory containing the train datasets.")
    parser.add_argument("--test_dir", type=str, default="data/acceptance_datasets/test", help="Path of the directory containing the test datasets.")
    parser.add_argument("--final_step", type=int, default=19725, help="Step to assign to the final model.")
    parser.add_argument("--results_dir", type=str, default="data/results", help="The directory where to save the tabular results.")
    parser.add_argument("--plots_dir", type=str, default="plots/probing", help="The directory where to save the plots.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = os.path.join(args.models_dir, args.model_name)
    checkpoints = get_sorted_checkpoints(model_dir)
    checkpoints.append((args.final_step, model_dir))

    tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")

    records = []
    fold_records = []
    for checkpoint, checkpoint_path in checkpoints:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, output_hidden_states=True)
        model.eval().to(device)

        for file_name in os.listdir(args.train_dir):
            dataset_name = file_name.split(".")[0]
            if "coherence" in dataset_name:
                continue
            train_sentences, train_labels, _ = load_split(args.train_dir, file_name)
            test_sentences, test_labels, test_df = load_split(args.test_dir, file_name)
            test_categories = get_test_categories(dataset_name, test_df)

            train_embeddings = get_last_token_embeddings(model, train_sentences, tokenizer, device).numpy()
            test_embeddings = get_last_token_embeddings(model, test_sentences, tokenizer, device).numpy()

            results = probe_all_layers(train_embeddings, train_labels, test_embeddings, test_labels, test_categories)

            if "fold" not in dataset_name:
                for r in results:
                    records.append({
                        "checkpoint": checkpoint,
                        "dataset": dataset_name,
                        "model": args.model_name,
                        "layer": r["layer"],
                        "category": r["category"],
                        "accuracy": r["accuracy"],
                        "accuracy_std": 0.0,
                        "f1": r["f1"],
                        "f1_std": 0.0,
                    })
            else: 
                fold = int((file_name.split(".")[0]).split("_")[-1])
                for r in results:
                    fold_records.append({
                        "checkpoint": checkpoint,
                        "dataset": "temporal_concord",
                        "model": args.model_name,
                        "layer": r["layer"],
                        "category": r["category"],
                        "fold": fold,
                        "accuracy": r["accuracy"],
                        "f1": r["f1"],
                    })

        if fold_records:
            fold_df = pd.DataFrame(fold_records)
            agg = fold_df.groupby(["checkpoint", "dataset", "model", "layer", "category"]).agg(
                accuracy=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                f1=("f1", "mean"),
                f1_std=("f1", "std"),
            ).reset_index()
            records.extend(agg.to_dict("records"))

        del model
        torch.cuda.empty_cache()

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    df = pd.DataFrame(records, columns=["checkpoint", "dataset", "model", "layer", "category",
                                         "accuracy", "accuracy_std", "f1", "f1_std"])
    df.to_csv(os.path.join(args.results_dir, f"probing_{args.model_name}_checkpoints.csv"), index=False)

    if not os.path.exists(args.plots_dir):
        os.makedirs(args.plots_dir)

    global_df = df[df["category"] == "all"]
    plot_checkpoint_scores(global_df, args.model_name, os.path.join(args.plots_dir, f"probing_{args.model_name}_checkpoints_prova.svg"))


if __name__ == "__main__":
    main()
