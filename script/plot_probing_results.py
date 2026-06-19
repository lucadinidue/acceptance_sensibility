from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os

from utils import sort_model_names

STEPS_PER_EPOCH = 19725 / 3


# def _checkpoint_fmt(x, _):
#     return f"{x / 1000:.0f}k"

def _checkpoint_fmt(x, _):
    if x >= 1000:
        return f"{x/1000:g}k"
    return f"{int(x)}"


def _style_checkpoint_ax(ax, ylabel):
    for i in (1, 2):
        ax.axvline(i * STEPS_PER_EPOCH, color="#888888",
                   linestyle="--", linewidth=1.0, alpha=0.7, zorder=0)
    ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(FuncFormatter(_checkpoint_fmt))


def _style_plain_ax(ax, xlabel, ylabel):
    ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _plot_line_with_band(ax, x, y, std, color, label):
    ax.plot(x, y, color=color, marker="o", linewidth=1.9, markersize=3.5,
            alpha=0.9, label=label)
    if (std > 0).any():
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.15)


def _metric_info(metric):
    ylabel = "Probing F1" if metric == "f1" else "Probing accuracy"
    suffix = f"_{metric}" if metric != "f1" else ""
    return metric, f"{metric}_std", ylabel, suffix


def plot_last_layer_global(df, metric, plots_dir):
    val_col, std_col, ylabel, suffix = _metric_info(metric)

    df = df[df["layer"] == df.groupby("model")["layer"].transform("max")]
    df = df[df["category"] == "all"]

    dataset_names = sorted(df["dataset"].unique().tolist())
    n_datasets = len(dataset_names)
    hue_order = sort_model_names(df["model"].unique().tolist())
    palette = sns.color_palette("tab10", n_colors=len(hue_order))
    color_map = dict(zip(hue_order, palette))

    fig, axes = plt.subplots(nrows=1, ncols=n_datasets, figsize=(5.4 * n_datasets, 4.2))
    if n_datasets == 1:
        axes = [axes]

    for ax, dataset_name in zip(axes, dataset_names):
        subset = df[df["dataset"] == dataset_name]
        for model_name in hue_order:
            model_data = subset[subset["model"] == model_name].sort_values("checkpoint")
            if model_data.empty:
                continue
            _plot_line_with_band(ax, model_data["checkpoint"], model_data[val_col],
                                 model_data[std_col], color_map[model_name], model_name)
        _style_checkpoint_ax(ax, ylabel)
        ax.set_title(dataset_name, fontsize=13)

    handles = [
        plt.Line2D([0], [0], color=color_map[m], marker="o", linewidth=1.9, markersize=3.5, label=m)
        for m in hue_order
    ]
    fig.suptitle("Linear probing across checkpoints (last layer)", fontsize=16)
    fig.legend(handles, hue_order, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])

    fig.savefig(os.path.join(plots_dir, f"probing_checkpoints_last_layer_global{suffix}.svg"), dpi=220)
    plt.close(fig)


def plot_last_layer_categories(df, metric, plots_dir):
    val_col, std_col, ylabel, suffix = _metric_info(metric)

    df = df[df["layer"] == df.groupby("model")["layer"].transform("max")]
    df = df[df["category"] != "all"]

    for model_name in sort_model_names(df["model"].unique().tolist()):
        model_df = df[df["model"] == model_name]
        dataset_names = sorted(model_df["dataset"].unique().tolist())
        n_datasets = len(dataset_names)

        fig, axes = plt.subplots(nrows=1, ncols=n_datasets, figsize=(5.4 * n_datasets, 4.2))
        if n_datasets == 1:
            axes = [axes]

        for ax, dataset_name in zip(axes, dataset_names):
            subset = model_df[model_df["dataset"] == dataset_name]
            categories = sorted(subset["category"].unique().tolist())
            palette = sns.color_palette("tab10", n_colors=len(categories))
            color_map = dict(zip(categories, palette))
            for cat in categories:
                cat_data = subset[subset["category"] == cat].sort_values("checkpoint")
                if cat_data.empty:
                    continue
                _plot_line_with_band(ax, cat_data["checkpoint"], cat_data[val_col],
                                     cat_data[std_col], color_map[cat], cat)
            _style_checkpoint_ax(ax, ylabel)
            ax.set_title(dataset_name, fontsize=13)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                      ncol=2, fontsize=7, frameon=False, borderaxespad=0.)

        fig.suptitle(f"Linear probing across checkpoints — {model_name}", fontsize=14)

        safe_model = model_name.replace("/", "_")
        output_path = os.path.join(plots_dir, f"probing_checkpoints_last_layer_categories_{safe_model}{suffix}.svg")
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_all_layers_global(df, metric, plots_dir):
    val_col, _, ylabel, suffix = _metric_info(metric)

    df = df[df["category"] == "all"]
    cmap = sns.color_palette("crest", as_cmap=True)

    for model_name in sort_model_names(df["model"].unique().tolist()):
        model_df = df[df["model"] == model_name]
        dataset_names = sorted(model_df["dataset"].unique().tolist())
        n_datasets = len(dataset_names)
        layers = sorted(model_df["layer"].unique().tolist())
        vmax_layer = layers[-1] if len(layers) > 1 else layers[0] + 1
        norm = Normalize(vmin=layers[0], vmax=vmax_layer)

        fig, axes = plt.subplots(nrows=1, ncols=n_datasets, figsize=(5.4 * n_datasets, 4.2),
                                 constrained_layout=True)
        axes = [axes] if n_datasets == 1 else list(axes)

        for ax, dataset_name in zip(axes, dataset_names):
            subset = model_df[model_df["dataset"] == dataset_name]
            for layer_idx in layers:
                layer_data = subset[subset["layer"] == layer_idx].sort_values("checkpoint")
                if layer_data.empty:
                    continue
                color = cmap(norm(layer_idx))
                ax.plot(layer_data["checkpoint"], layer_data[val_col],
                        color=color, linewidth=1.5, alpha=0.85)
            _style_checkpoint_ax(ax, ylabel)
            ax.set_title(dataset_name, fontsize=13)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, location="right", fraction=0.025, pad=0.02, label="Layer")
        fig.suptitle(f"Linear probing across checkpoints (all layers) — {model_name}", fontsize=14)

        safe_model = model_name.replace("/", "_")
        fig.savefig(os.path.join(plots_dir, f"probing_checkpoints_all_layers_global_{safe_model}{suffix}.svg"), dpi=220)
        plt.close(fig)


def plot_layer_profile(df, metric, checkpoint_arg, normalize_layers, plots_dir):
    val_col, std_col, ylabel, suffix = _metric_info(metric)

    df = df[df["category"] == "all"]
    target_ck = checkpoint_arg if checkpoint_arg is not None else int(df["checkpoint"].max())

    models = sort_model_names(df["model"].unique().tolist())
    palette = sns.color_palette("tab10", n_colors=len(models))
    color_map = dict(zip(models, palette))

    # Precompute per-model layer ranges for normalization.
    model_layer_range = {
        m: (df[df["model"] == m]["layer"].min(), df[df["model"] == m]["layer"].max())
        for m in models
    }

    dataset_names = sorted(df["dataset"].unique().tolist())
    n_datasets = len(dataset_names)

    fig, axes = plt.subplots(nrows=1, ncols=n_datasets, figsize=(5.4 * n_datasets, 4.2))
    if n_datasets == 1:
        axes = [axes]

    for ax, dataset_name in zip(axes, dataset_names):
        subset = df[df["dataset"] == dataset_name]
        for model_name in models:
            model_data = subset[subset["model"] == model_name]
            if model_data.empty:
                continue
            used_ck = target_ck if target_ck in model_data["checkpoint"].values else int(model_data["checkpoint"].max())
            ck_data = model_data[model_data["checkpoint"] == used_ck].sort_values("layer")
            if ck_data.empty:
                continue
            x = ck_data["layer"]
            if normalize_layers:
                l_min, l_max = model_layer_range[model_name]
                x = (x - l_min) / (l_max - l_min) if l_max > l_min else x * 0.0
            _plot_line_with_band(ax, x, ck_data[val_col],
                                 ck_data[std_col], color_map[model_name], model_name)
        xlabel = "Relative depth" if normalize_layers else "Layer"
        _style_plain_ax(ax, xlabel, ylabel)
        ax.set_title(dataset_name, fontsize=13)

    handles = [
        plt.Line2D([0], [0], color=color_map[m], marker="o", linewidth=1.9, markersize=3.5, label=m)
        for m in models
    ]
    ck_label = f"{target_ck / 1000:.0f}k" if target_ck >= 1000 else str(target_ck)
    fig.suptitle(f"Layer profile at checkpoint {ck_label}", fontsize=16)
    fig.legend(handles, models, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])

    norm_suffix = "_normlayers" if normalize_layers else ""
    fig.savefig(os.path.join(plots_dir, f"probing_checkpoints_layer_profile_ck{target_ck}{norm_suffix}{suffix}.svg"), dpi=220)
    plt.close(fig)


def plot_heatmap_layers_checkpoints(df, metric, plots_dir):
    val_col, _, cb_label, suffix = _metric_info(metric)

    df = df[df["category"] == "all"]
    cmap = sns.color_palette("crest", as_cmap=True)

    for model_name in sort_model_names(df["model"].unique().tolist()):
        model_df = df[df["model"] == model_name]
        dataset_names = sorted(model_df["dataset"].unique().tolist())
        n_datasets = len(dataset_names)
        vmin, vmax = model_df[val_col].min(), model_df[val_col].max()

        fig, axes = plt.subplots(nrows=1, ncols=n_datasets, figsize=(5.4 * n_datasets, 4.2))
        if n_datasets == 1:
            axes = [axes]

        pcm = None
        for ax, dataset_name in zip(axes, dataset_names):
            subset = model_df[model_df["dataset"] == dataset_name]
            pivot = subset.pivot_table(index="layer", columns="checkpoint", values=val_col)
            pcm = ax.pcolormesh(pivot.columns.values, pivot.index.values, pivot.values,
                                cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
            ax.set_title(dataset_name, fontsize=13)
            ax.set_xlabel("Checkpoint")
            ax.set_ylabel("Layer")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.xaxis.set_major_formatter(FuncFormatter(_checkpoint_fmt))

        if pcm is not None:
            fig.colorbar(pcm, ax=axes, label=cb_label)
        fig.suptitle(f"Layer × Checkpoint heatmap — {model_name}", fontsize=14)
        fig.tight_layout()

        safe_model = model_name.replace("/", "_")
        fig.savefig(os.path.join(plots_dir, f"probing_checkpoints_heatmap_layers_checkpoints_{safe_model}{suffix}.svg"), dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Linear probing across pretraining checkpoints.")
    parser.add_argument("--results_dir", type=str, default="data/results/probing")
    parser.add_argument("--plots_dir", type=str, default="plots/probing")
    parser.add_argument("--mode", type=str, choices=["last_layer_global", "last_layer_categories", 
                        "all_layers_global", "layer_profile", "heatmap_layers_checkpoints"],
                        default="last_layer_global")
    parser.add_argument("--metric", type=str, choices=["f1", "accuracy"], default="f1")
    parser.add_argument("--checkpoint", type=int, default=None, help="Fixed checkpoint for layer_profile mode.")
    parser.add_argument("--normalize_layers", action="store_true", default=False,
                        help="Normalize layer indices to [0, 1] per model (layer_profile only).")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    all_results = []
    for file_name in os.listdir(args.results_dir):
        if not file_name.endswith("_checkpoints.csv"):
            continue
        model_name = file_name[len("probing_gpt_"):-len("_42s_checkpoints.csv")]
        results_path = os.path.join(args.results_dir, file_name)
        res_df = pd.read_csv(results_path)
        res_df["model"] = model_name
        all_results.append(res_df)

    df = pd.concat(all_results)

    if args.mode == "last_layer_global":
        plot_last_layer_global(df, args.metric, args.plots_dir)
    elif args.mode == "last_layer_categories":
        plot_last_layer_categories(df, args.metric, args.plots_dir)
    elif args.mode == "all_layers_global":
        plot_all_layers_global(df, args.metric, args.plots_dir)
    elif args.mode == "layer_profile":
        plot_layer_profile(df, args.metric, args.checkpoint, args.normalize_layers, args.plots_dir)
    elif args.mode == "heatmap_layers_checkpoints":
        plot_heatmap_layers_checkpoints(df, args.metric, args.plots_dir)


if __name__ == "__main__":
    main()
