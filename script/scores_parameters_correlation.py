from matplotlib.ticker import FuncFormatter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import argparse
import json
import os
import re

from scipy.stats import spearmanr


from plot_probing_results import (
    _style_checkpoint_ax,
    _metric_info,
    _checkpoint_fmt,
)

MODEL_RE = re.compile(r"gpt_(\d+)l_(\d+)h_(\d+)d_(\d+)s")


def load_and_parse(results_dir):
    frames = []
    for file_name in sorted(os.listdir(results_dir)):
        if not file_name.endswith(".csv"):
            continue
        frames.append(pd.read_csv(os.path.join(results_dir, file_name)))
    df = pd.concat(frames, ignore_index=True)

    parsed = df["model"].str.extract(MODEL_RE)
    df["n_layers"] = parsed[0].astype(int)
    df["n_heads"] = parsed[1].astype(int)
    df["hidden"] = parsed[2].astype(int)
    df["seed"] = parsed[3].astype(int)

    return df[df["category"] == "all"].copy()

def reduce_layer(df, reduction, val_col):
    if reduction == "last":
        last_layer = df.groupby("model")["layer"].transform("max")
        return df[df["layer"] == last_layer].copy()
    elif reduction == "first":
        return df[df["layer"] == 1].copy()
    elif reduction == "mean":
        last_layer = df.groupby("model")["layer"].transform("max")
        mean_layer = last_layer // 2
        return df[df["layer"] == mean_layer].copy()
    elif reduction == "max":
        idx = df.groupby(["model", "dataset", "checkpoint"])[val_col].idxmax()
        return df.loc[idx].copy()
    else:
        raise Exception(f"Reduction {reduction} not implemented")

def define_groups(df):
    def by_query(mask):
        return sorted(df.loc[mask, "model"].unique().tolist())

    depth_models = by_query((df["n_heads"] == 8) & (df["hidden"] == 512))
    heads_models = by_query((df["n_layers"] == 8) & (df["hidden"] == 512))
    shape_models = [
        "gpt_6l_9h_576d_42s",
        "gpt_8l_8h_512d_42s",
        "gpt_11l_7h_448d_42s",
        "gpt_16l_6h_384d_42s",
    ]

    return [
        {
            "name": "depth",
            "models": depth_models,
            "axis_col": "n_layers",
            "axis_label": "number of layers",
            "caption": "heads=8, hidden=512 fixed; depth varies",
        },
        {
            "name": "heads",
            "models": heads_models,
            "axis_col": "n_heads",
            "axis_label": "number of heads",
            "caption": "layers=8, hidden=512 fixed; heads vary",
        },
        {
            "name": "shape",
            "models": [m for m in shape_models if m in set(df["model"])],
            "axis_col": "n_layers",
            "axis_label": "number of layers",
            "caption": "~40M budget fixed; deeper = narrower (hidden & heads vary inversely)",
        },
    ]



def _maybe_log_x(ax, log_x):
    if not log_x:
        return
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_checkpoint_fmt))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())



def plot_group_raw(df, group, plots_dir, suffix, val_col, ylabel, plot_tag, log_x=False):
    sub = df[df["model"].isin(group["models"])]
    datasets = sorted(sub["dataset"].unique().tolist())
    n_datasets = len(datasets)
    axis_col = group["axis_col"]

    axis_values = sorted(sub[axis_col].unique().tolist())
    cmap = sns.color_palette("crest", as_cmap=True)
    vmin, vmax = min(axis_values), max(axis_values)
    norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1)

    fig, axes = plt.subplots(nrows=1, ncols=n_datasets,
                             figsize=(5.4 * n_datasets, 4.2), constrained_layout=True)
    axes = [axes] if n_datasets == 1 else list(axes)

    for ax, dataset in zip(axes, datasets):
        ds = sub[sub["dataset"] == dataset]
        for model_name in group["models"]:
            md = ds[ds["model"] == model_name].sort_values("checkpoint")
            if md.empty:
                continue
            color = cmap(norm(md[axis_col].iloc[0]))
            ax.plot(md["checkpoint"], md[val_col], color=color,
                    marker="o", linewidth=1.7, markersize=3.0, alpha=0.9)
        _style_checkpoint_ax(ax, ylabel)
        _maybe_log_x(ax, log_x)
        ax.set_title(dataset, fontsize=13)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, location="right", fraction=0.025, pad=0.02,
                 label=group["axis_label"])
    fig.suptitle(f"Last-layer score by {group['axis_label']} — group '{group['name']}'\n"
                 f"{group['caption']}", fontsize=12)

    out_path = os.path.join(plots_dir, f"correlation_{group['name']}_raw{suffix}_{plot_tag}.svg")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        "Correlate probing scores with architectural hyperparameters.")
    parser.add_argument("--results_dir", type=str, default="data/results/probing")
    parser.add_argument("--pretrained_dir", type=str, default="models/pretrained")
    parser.add_argument("--plots_dir", type=str, default="plots/probing_correlation")
    parser.add_argument("--params_cache", type=str, default="model_n_params.json")
    parser.add_argument("--metric", type=str, choices=["f1", "accuracy"], default="f1")
    parser.add_argument("--layer_reduction", type=str, choices=["last", "max", "mean", "first"], default="last")
    parser.add_argument("--raw", action="store_true", default=False,
                        help="Also emit the per-model raw-score inspection figures.")
    parser.add_argument("--log_x", action="store_true", default=False,
                        help="Use a logarithmic scale for the checkpoint (x) axis.")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)
    val_col, _, ylabel, suffix = _metric_info(args.metric)

    df = load_and_parse(args.results_dir)
    df = reduce_layer(df, args.layer_reduction, val_col)
    groups = define_groups(df)

    for group in groups:
        plot_group_raw(df, group, args.plots_dir, suffix, val_col, ylabel, args.layer_reduction, args.log_x)


if __name__ == "__main__":
    main()
