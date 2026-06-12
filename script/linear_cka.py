import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from utils import *
import numpy as np
import argparse
import os

sns.set_theme(style="whitegrid", font_scale=0.9)


def linear_cka_with_labels(X: np.ndarray, labels: np.ndarray) -> float:
    Y = np.eye(2)[labels.astype(int)]          # one-hot (n, 2)
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return num / den


def compute_cka_per_layer(embeddings: np.ndarray, labels: np.ndarray) -> list[float]:
    num_layers = embeddings.shape[1]
    scores = [linear_cka_with_labels(embeddings[:, l, :], labels) for l in range(num_layers)]
    return scores


def main():
    parser = argparse.ArgumentParser("CKA between representations and acceptability labels.")
    parser.add_argument("--models_dir", type=str, default="models/pretrained", help="The name of the model to analyze.")
    parser.add_argument("--results_dir", type=str, default="data/results", help="The directory where to save the tabular results.")
    parser.add_argument("--plots_dir", type=str, default="plots/cka", help="The directory where to save the plots.")
    parser.add_argument("--representations_dir", type=str, default="data/representations", help="Path of the directory containing the extracted representations.")
    parser.add_argument("--dataset_dir", type=str, default="data/acceptance_datasets/clean", help="Path of the directory containing the datasets.")
    args = parser.parse_args()

    cka_scores = []

    # with tqdm(total=(len(os.listdir(args.dataset_dir))-1)*len(os.listdir(args.models_dir))) as pbar:
    #     for dataset_file_name in os.listdir(args.dataset_dir):
    #         dataset_name = dataset_file_name.split(".")[0]
    #         if "coherence" in dataset_name:
    #             continue
    #         dataset_path = os.path.join(args.dataset_dir, dataset_file_name)

    #         for model_name in os.listdir(args.models_dir):
    #             representations_path = os.path.join(args.representations_dir, f"{dataset_name}_{model_name}.h5")

    #             ids, embeddings = load_representations(representations_path, model_name, dataset_name)
    #             acceptability_labels = load_acceptability_labels(dataset_path, ids)

    #             scores = compute_cka_per_layer(embeddings, acceptability_labels)
    #             for layer, score in enumerate(scores):
    #                 cka_scores.append({"dataset": dataset_name, "model": model_name[4:-4], "layer": layer, "score": score}) #layer/(len(scores)-1)
    #             pbar.update(1)
    
    # df = pd.DataFrame(cka_scores)
    # df.to_csv(os.path.join(args.results_dir, f"cka_training_end.csv"))
    df = pd.read_csv(os.path.join(args.results_dir, f"cka_training_end.csv"))
    hue_order = sort_model_names(df['model'].unique().tolist())
    plot_scores(df, hue_order, title="Linear CKA", ylabel="CKA score", output_path=os.path.join(args.plots_dir, "cka_training_end.svg"))


if __name__ == "__main__":
    main()