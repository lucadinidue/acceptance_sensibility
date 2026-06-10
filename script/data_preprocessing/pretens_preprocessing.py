import pandas as pd
import json


def main():
    src_path = "data/acceptance_datasets/src/pretens-It-Subtask1-labels.tsv"
    out_path = "data/acceptance_datasets/clean/pretens_it.jsonl"

    df = pd.read_csv(src_path, sep="\t")
    
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "id": f"PT_{row['ID']}",
                "source": "PT",
                "sentence": row["Sentence"],
                "acceptability": int(row["Labels"]),
                "metadata": row["Construction"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()