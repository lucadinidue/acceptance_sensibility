import pandas as pd
import json


def main():
    src_path = "data/acceptance_datasets/src/Blimp_It.csv"
    out_path = "data/acceptance_datasets/clean/blimp_it.jsonl"

    df = pd.read_csv(src_path)
    df = df[df["Inspiring source for the phenomenon"] != "COnVERSA"]

    with open(out_path, "w", encoding="utf-8") as f:
        for row_idx, row in df.iterrows():
            metadata = {
                "macro_phenomenon": row["Macro-phenomenon"],
                "micro_phenomenon": row["Micro-phenomenon"],
            }

            for sent_idx, (sentence, acceptability) in enumerate([
                (row["Good Generated Sentence"], 1),
                (row["Bad Generated Sentence"],  0),
            ]):
                sent_id = 2*row_idx + sent_idx
                record = {
                    "id": f"BL_{sent_id}",
                    "sentence": sentence,
                    "acceptability": acceptability,
                    "source": "blimp_it",
                    "metadata": metadata
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()