from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os


def pairs_to_records(pairs_df):
    records = []
    for row_idx, row in pairs_df.iterrows():
        metadata = {
            "macro_phenomenon": row["Macro-phenomenon"],
            "micro_phenomenon": row["Micro-phenomenon"],
        }
        for sent_idx, (sentence, acceptability) in enumerate([
            (row["Good Generated Sentence"], 1),
            (row["Bad Generated Sentence"],  0),
        ]):
            sent_id = 2 * row_idx + sent_idx
            records.append({
                "id": f"BL_{sent_id}",
                "sentence": sentence,
                "acceptability": acceptability,
                "source": "blimp_it",
                "metadata": metadata,
            })
    return records

def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    src_path = "data/acceptance_datasets/src/Blimp_It.csv"
    out_dir = "data/acceptance_datasets"

    df = pd.read_csv(src_path)
    df = df[df["Inspiring source for the phenomenon"] != "COnVERSA"]

    train_pairs, test_pairs = train_test_split(df, test_size=0.2, stratify=df["Micro-phenomenon"],random_state=42)

    train_records = pairs_to_records(train_pairs)
    test_records  = pairs_to_records(test_pairs)

    write_jsonl(os.path.join(out_dir, "clean_merged", "blimp_it.jsonl"), pairs_to_records(df))
    write_jsonl(os.path.join(out_dir, "train", "blimp_it.jsonl"), train_records)
    write_jsonl(os.path.join(out_dir, "test", "blimp_it.jsonl"),  test_records)

if __name__ == "__main__":
    main()