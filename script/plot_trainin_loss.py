from utils import sort_model_names
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import json
import os

sns.set_theme(style="whitegrid", font_scale=0.9)

def plot_training_losses(df, output_path):
    df["model"] = df["model"].str.slice(4, -4)
    hue_order = sort_model_names(df['model'].unique().tolist())
    sns.lineplot(df, x="step", y="loss", hue="model", hue_order=hue_order)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    

def load_log_history(trainer_state_path):
    with open(trainer_state_path, 'r') as src_file:
        trainer_state = json.load(src_file)
    return trainer_state["log_history"]


def main():
    parser = argparse.ArgumentParser("Plotting training loss.")
    parser.add_argument("--models_dir", type=str, default="models/pretrained", help="Directory containing pretrained models.")
    parser.add_argument("--output_path", type=str, default="plots/trainig_losses.svg", help="Path where to save the plot.")
    args = parser.parse_args()

    records = []
    for model_name in os.listdir(args.models_dir):
        trainer_state_path = os.path.join(args.models_dir, model_name, "trainer_state.json")
        log_history = load_log_history(trainer_state_path)
        for step_log in log_history:
            if "loss" in step_log.keys():
                records.append({
                    "model": model_name,
                    "step": step_log["step"],
                    "loss": step_log["loss"]
                })
    loss_df = pd.DataFrame.from_records(records)
    plot_training_losses(loss_df, args.output_path)

if __name__ == "__main__":
    main()
