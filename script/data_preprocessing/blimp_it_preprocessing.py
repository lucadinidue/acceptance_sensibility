import pandas as pd
import json


def main():
    src_path = "data/acceptance_datasets/src/Blimp_It.csv"
    out_path = "data/acceptance_datasets/clean/blimp_it.jsonl"

    df = pd.read_csv(src_path)
    df = df[df["Inspiring source for the phenomenon"] != "COnVERSA"]

    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            pair_num = str(row["Pair Number "]).zfill(3)
            metadata = {
                "macro_phenomenon": row["Macro-phenomenon"],
                "micro_phenomenon": row["Micro-phenomenon"],
            }

            for label, sentence, acceptability in [
                ("good", row["Good Generated Sentence"], 1),
                ("bad",  row["Bad Generated Sentence"],  0),
            ]:
                record = {
                    "id": f"BL_{pair_num}_{label}",
                    "sentence": sentence,
                    "acceptability": acceptability,
                    "source": "blimp_it",
                    "metadata": metadata
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()