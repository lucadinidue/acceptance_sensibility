
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from utils import *
import numpy as np
import argparse
import os

sns.set_theme(style="whitegrid", font_scale=0.9)

def prepare_layer_data(embeddings, labels, layer, test_size=0.2, seed=42):
    X = embeddings[:, layer, :]            # (N, H)
    y = np.asarray(labels).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,            # mantiene le proporzioni delle classi
        random_state=seed,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def train_probe(X_train, y_train, C=1.0, seed=42):
    probe = RidgeClassifier(
        class_weight="balanced",
    )
    probe.fit(X_train, y_train)
    return probe


def probe_layer(embeddings, labels, layer, test_size=0.2, seed=42):
    X_train, X_test, y_train, y_test = prepare_layer_data(embeddings, labels, layer, test_size, seed)
    probe = train_probe(X_train, y_train, seed=seed)
    y_pred = probe.predict(X_test)

    return {
        "layer": layer,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def probe_all_layers(embeddings, labels, test_size=0.2, seed=42):
    num_layers = embeddings.shape[1]
    return [
        probe_layer(embeddings, labels, layer, test_size, seed)
        for layer in range(num_layers)
    ]


def main():
    parser = argparse.ArgumentParser("Compute CKA between acceptable and not acceptable sentences representations.")
    parser.add_argument("--models_dir", type=str, default="models/pretrained", help="The name of the model to analyze.")
    parser.add_argument("--results_dir", type=str, default="data/results", help="The directory where to save the tabular results.")
    parser.add_argument("--plots_dir", type=str, default="plots/probing", help="The directory where to save the plots.")
    parser.add_argument("--representations_dir", type=str, default="data/representations", help="Path of the directory containing the extracted representations.")
    parser.add_argument("--dataset_dir", type=str, default="data/acceptance_datasets/clean", help="Path of the directory containing the datasets.")
    args = parser.parse_args()

    records = []
    for dataset_file_name in os.listdir(args.dataset_dir):
        if 'coherence' in dataset_file_name:
            continue
        dataset_name = dataset_file_name.split(".")[0]
        dataset_path = os.path.join(args.dataset_dir, dataset_file_name)

        for model_name in tqdm(os.listdir(args.models_dir)):
            representations_path = os.path.join(args.representations_dir, f"{dataset_name}_{model_name}.h5")
            ids, embeddings = load_representations(representations_path, model_name, dataset_name)
            labels = load_acceptability_labels(dataset_path, ids)
            results = probe_all_layers(embeddings, labels)

            for r in results:
                records.append({"dataset": dataset_name,"model": model_name, "layer": r["layer"], "score": r["accuracy"], "f1": r["f1"],})

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.results_dir, f"probing_training_end.csv"), index=False)
    # df = pd.read_csv(os.path.join(args.results_dir, f"probing_training_end.csv"))
    plot_scores(df, title="Linear Probing", ylabel="Probing accuracy", output_path=os.path.join(args.plots_dir, "probing_training_end.svg"))


if __name__ == "__main__":
    main()