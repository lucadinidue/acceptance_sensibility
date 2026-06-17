import pandas as pd
import numpy as np
import json
import os


def subsample_dataset(df):
    rng = np.random.default_rng(42)
    df = df.copy()

    df['perturbation_type'] = np.select = np.where(
        df['label'] == 'Orig', 'orig',
        np.where(df['label'].str.startswith('sub'), 'sub', 'swap')
    )

    rows = []
    for _, group in df.groupby('passage_id', sort=False):
        orig = group[group['perturbation_type'] == 'orig']
        rows.append(orig.iloc[0])

        sampled_perturbation = rng.choice(['sub', 'swap'])
        candidates = group[group['perturbation_type'] == sampled_perturbation]
        sampled_idx = rng.choice(candidates.index)
        rows.append(df.loc[sampled_idx])

    result = pd.DataFrame(rows).drop(columns='perturbation_type').reset_index(drop=True)
    return result


def write_df_to_file(df, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "id": f"CO_{row['passage_id']}_{row['label']}",
                "source": "CO_WIKI",
                "sentence": row["text"],
                "acceptability": 1 if row["label"] == "Orig" else 0,
                "metadata": row["label"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    train_path = "data/acceptance_datasets/src/coherence_wiki_it_train.tsv"
    eval_path = "data/acceptance_datasets/src/coherence_wiki_it_eval.tsv"
    out_dir = "data/acceptance_datasets/"

    train_df = pd.read_csv(train_path, sep="\t")
    train_df = subsample_dataset(train_df)
    train_path = os.path.join(out_dir, "train", "coherence_it.jsonl")
    write_df_to_file(train_df, train_path)

    test_df = pd.read_csv(eval_path, sep="\t")
    test_df = subsample_dataset(test_df)
    test_path = os.path.join(out_dir, "test", "coherence_it.jsonl")
    write_df_to_file(test_df, test_path)

    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    merged_path = os.path.join(out_dir, "clean_merged", "coherence_it.jsonl")
    write_df_to_file(merged_df, merged_path)
    

if __name__ == "__main__":
    main()