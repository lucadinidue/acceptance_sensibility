from scipy.interpolate import RectBivariateSpline
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import os
import re

STEPS_PER_EPOCH = 19725 / 3


def _metric_info(metric):
    ylabel = "Probing F1" if metric == "f1" else "Probing accuracy"
    suffix = f"_{metric}" if metric != "f1" else ""
    return metric, f"{metric}_std", ylabel, suffix

def _checkpoint_fmt(x, _):
    if x >= 1000:
        return f"{x/1000:g}k"
    return f"{int(x)}"


def _reduce_layers(df, val_col, layer_mode):
    """Seleziona una riga per (dataset, model, checkpoint):
       - 'last': l'ultimo layer di ciascun modello
       - 'best': il layer che massimizza val_col (max-over-layers)"""
    df = df[df["category"] == "all"].copy()
    if layer_mode == "last":
        return df[df["layer"] == df.groupby("model")["layer"].transform("max")]
    if layer_mode == "best":
        idx = df.groupby(["dataset", "model", "checkpoint"])[val_col].idxmax()
        return df.loc[idx]
    raise ValueError(f"layer_mode sconosciuto: {layer_mode!r} (usa 'last' o 'best')")


def plot_layer_checkpoints(df, metric, plots_dir, layer_mode="last"):
    val_col, std_col, ylabel, suffix = _metric_info(metric)

    df = _reduce_layers(df, val_col, layer_mode)
    df["layer_counts"] = df["model"].apply(lambda x: int(re.search(r"(\d+)l", x).group(1)))

    dataset_names = sorted(df["dataset"].unique().tolist())

    fig, axes = plt.subplots(nrows=1, ncols=len(dataset_names), figsize=(5.4 * len(dataset_names), 4.2), constrained_layout=True)
    axes = [axes] if len(dataset_names) == 1 else list(axes)
    cmap = sns.color_palette("RdYlGn", as_cmap=True)  
    for ax, dataset_name in zip(axes, dataset_names):
        dataset_df = df[df["dataset"] == dataset_name]
        pivot = (dataset_df.pivot_table(index="layer_counts", columns="checkpoint", values=val_col, aggfunc="mean")
                 .sort_index(axis=0).sort_index(axis=1))

        layer_vals = pivot.index.values.astype(float)
        checkpoints = pivot.columns.values.astype(float)
        Z_raw = pivot.values.astype(float)      
        Z = Z_raw.copy()

        for i in range(Z.shape[0]):
            last_seen = np.nan
            for j in range(Z.shape[1]):
                if not np.isnan(Z[i, j]):
                    last_seen = Z[i, j]
                elif not np.isnan(last_seen):
                    Z[i, j] = last_seen
        col_means = np.nanmean(Z, axis=0)
        nan_pos = np.where(np.isnan(Z))
        Z[nan_pos] = np.take(col_means, nan_pos[1])

        vmin, vmax = np.nanmin(Z), np.nanmax(Z)

     
        ly_fine = np.linspace(layer_vals.min(), layer_vals.max(), 200)
        cx_fine = np.linspace(checkpoints.min(), checkpoints.max(), 300)
        spline = RectBivariateSpline(layer_vals, checkpoints, Z,
                                     kx=min(3, len(layer_vals) - 1),
                                     ky=min(3, len(checkpoints) - 1))
        grid = np.clip(spline(ly_fine, cx_fine), vmin, vmax)

        mesh = ax.pcolormesh(cx_fine, ly_fine, grid, cmap=cmap, vmin=vmin, vmax=vmax,
                             shading="auto", rasterized=True)

        levels = np.linspace(vmin, vmax, 7)[1:-1]
        cs = ax.contour(cx_fine, ly_fine, grid, levels=levels,
                        colors="black", linewidths=0.6, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=7.5, fmt="%.2f", inline_spacing=3)

        for i in (1, 2):
            ax.axvline(i * STEPS_PER_EPOCH, color="#ffffff", linestyle="--",
                       linewidth=1.0, alpha=0.6, zorder=3)

        for yi, lv in enumerate(layer_vals):
            for xj, ck in enumerate(checkpoints):
                if not np.isnan(Z_raw[yi, xj]):
                    ax.scatter(ck, lv, s=6, color="white", alpha=0.5, zorder=4, linewidths=0)

        ax.set_xlabel("checkpoint")
        ax.set_ylabel("number of layers")
        ax.set_title(dataset_name, fontsize=13)

        ax.set_yticks(layer_vals)
        xt_idx = np.linspace(0, len(checkpoints) - 1, min(7, len(checkpoints))).astype(int)
        ax.set_xticks(checkpoints[xt_idx])
        ax.xaxis.set_major_formatter(FuncFormatter(_checkpoint_fmt))
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label=ylabel)

    fig.suptitle(f"Linear probing across checkpoints and layers ({layer_mode} layer)", fontsize=16)
    fig.savefig(os.path.join(plots_dir, f"probing_checkpoints_{layer_mode}_layer_contour{suffix}.svg"),
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def aggregate_seed_scores(df):
    group_cols = ["model", "dataset", "layer", "category", "checkpoint"]
    agg = df.groupby(group_cols, as_index=False).agg(
        accuracy=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1=("f1", "mean"),
        f1_std=("f1", "std"),
    )
    agg["accuracy_std"] = agg["accuracy_std"].fillna(0.0)
    agg["f1_std"] = agg["f1_std"].fillna(0.0)
    return agg


def load_probing_results(results_dir):
    seed_re = re.compile(r"_(\d+)s_checkpoints\.csv$")

    all_results = []
    for file_name in os.listdir(results_dir):
        if not file_name.endswith("_checkpoints.csv"):
            continue
        seed_match = seed_re.search(file_name)

        model_name = file_name[len("probing_gpt_"):seed_match.start()]
        res_df = pd.read_csv(os.path.join(results_dir, file_name))
        res_df["model"] = model_name
        res_df["seed"] = int(seed_match.group(1))
        all_results.append(res_df)

    df = pd.concat(all_results, ignore_index=True)
    return aggregate_seed_scores(df)

def main():
    parser = argparse.ArgumentParser("Linear probing across pretraining checkpoints.")
    parser.add_argument("--results_dir", type=str, default="data/results/probing")
    parser.add_argument("--plots_dir", type=str, default="plots/probing")
    parser.add_argument("--metric", type=str, choices=["f1", "accuracy"], default="f1")
    parser.add_argument("--layer_mode", type=str, choices=["last", "best"], default="last")

    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    df = load_probing_results(args.results_dir)
    plot_layer_checkpoints(df, args.metric, args.plots_dir, args.layer_mode)
   


if __name__ == "__main__":
    main()