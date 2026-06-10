import pandas as pd
import json

def main():
    src_path = "data/acceptance_datasets/src/temporal-concord-annotated-cleaned-sentence.tsv"
    out_path = "data/acceptance_datasets/clean/temporal_concord.jsonl"

    df = pd.read_csv(src_path, sep="\t")
    df = df.rename(columns={"grammaticality": "acceptability"})
    

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            record = {
                "id": f"TC_{i:04d}",
                "source": "TC",
                "sentence": row["sentence"],
                "acceptability": int(row["acceptability"]),
                "metadata": row["placement"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()