from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import numpy as np
import json
import os

def build_records(df):
    records = []
    for i, row in df.iterrows():
        records.append({
            "id": f"TC_{i:04d}",
            "source": "TC",
            "sentence": row["sentence"],
            "acceptability": int(row["acceptability"]),
            "metadata": row["placement"],
        })
    return records

def infer_sentences_groups(df):
    group_ids = []
    group_id = 0
    past_tokens = set()

    for _, row in df.iterrows():
        tokens = set(row["sentence"].split())
        diff = len(tokens.symmetric_difference(past_tokens))
        if diff > 6:
            group_id += 1
        group_ids.append(group_id)
        past_tokens = tokens
    
    return np.array(group_ids)

def write_jsonl(path, records, idxs):
    with open(path, "w", encoding="utf-8") as f:
        for i in idxs:
            f.write(json.dumps(records[i], ensure_ascii=False) + "\n")

def main():
    src_path = "data/acceptance_datasets/src/temporal-concord-annotated-cleaned-sentence.tsv"
    out_dir = "data/acceptance_datasets/"

    df = pd.read_csv(src_path, sep="\t")
    df = df.rename(columns={"grammaticality": "acceptability"})
    labels = df["acceptability"].tolist()
    dataset_records = build_records(df)

    group_ids = infer_sentences_groups(df)

    dataset_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    X = np.zeros(len(group_ids))  # placeholder

    for fold, (train_idx, test_idx) in enumerate(dataset_splitter.split(X, labels, group_ids)):

        write_jsonl(os.path.join(out_dir, "train", f"temporal_concord_fold_{fold}.jsonl"),
                    dataset_records, train_idx)
        write_jsonl(os.path.join(out_dir, "test", f"temporal_concord_fold_{fold}.jsonl"),
                    dataset_records, test_idx)

    write_jsonl(os.path.join(out_dir, "clean_merged", "temporal_concord.jsonl"), dataset_records, range(len(dataset_records))
)

if __name__ == "__main__":
    main()