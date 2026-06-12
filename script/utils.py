import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import h5py
import re

sns.set_theme(style="whitegrid", font_scale=0.9)


def sort_model_names(model_names):
    def sort_key(name):
        layers = int(re.search(r'(\d+)l', name).group(1))
        heads  = int(re.search(r'(\d+)h', name).group(1))
        return (layers, heads)
    
    return sorted(model_names, key=sort_key)

def load_representations(representations_path, model_name, dataset_name):
    with h5py.File(representations_path, "r") as f:
        data = f[model_name][dataset_name]
        ids  = data["ids"].asstr()[:] 
        embeddings = data["last_token"][:]    # (N, num_layers, H)

    return ids, embeddings

def load_acceptability_labels(dataset_path, ids, is_mask=False):
    dataset = pd.read_json(dataset_path, lines=True).set_index("id")
    acceptability = dataset.loc[ids]['acceptability'].tolist()
    acceptability = np.array(acceptability, dtype=bool if is_mask else int)
    return acceptability


def plot_scores(df, hue_order, title, ylabel, output_path, score_col="score", dataset_titles=None, sharey=False):
    dataset_titles = dataset_titles or {}
    dataset_names = sorted(df["dataset"].unique().tolist())
    n_datasets = len(dataset_names)

    fig, axes = plt.subplots(nrows=1, ncols=n_datasets, figsize=(5.4 * n_datasets, 4.2), sharey=sharey)
    
    if n_datasets == 1:
        axes = [axes]

    palette = sns.color_palette("tab10", n_colors=len(hue_order))
    color_map = dict(zip(hue_order, palette))

    for ax, dataset_name in zip(axes, dataset_names):
        subset = df[df["dataset"] == dataset_name]

        for model_name in hue_order:
            model_data = subset[subset["model"] == model_name].sort_values("layer")

            # Normalizza la profondità dei layer tra 0 e 1
            layers = model_data["layer"].to_numpy()
            span = layers.max() - layers.min()
            norm_layers = (layers - layers.min()) / span if span > 0 else layers * 0.0

            ax.plot(norm_layers, model_data[score_col].to_numpy(), color=color_map[model_name], marker="o", linewidth=1.9, markersize=3.5,
                    alpha=0.9, label=model_name)

        ax.set_title(dataset_titles.get(dataset_name, dataset_name), fontsize=13)
        ax.set_xlabel("Normalized layer depth (0=input, 1=final)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Shared legend
    handles = [plt.Line2D([0], [0], color=color_map[m], marker="o", linewidth=1.9, markersize=3.5, label=m) for m in hue_order]
    fig.suptitle(title, fontsize=16)
    fig.legend(handles, hue_order, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02),)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)