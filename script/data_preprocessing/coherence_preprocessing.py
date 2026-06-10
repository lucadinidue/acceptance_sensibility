import pandas as pd
import json


def main():
    train_path = "data/acceptance_datasets/src/coherence_wiki_it_train.tsv"
    eval_path = "data/acceptance_datasets/src/coherence_wiki_it_eval.tsv"
    out_path = "data/acceptance_datasets/clean/coherence_it.jsonl"

    df_train = pd.read_csv(train_path, sep="\t")
    df_eval = pd.read_csv(eval_path, sep="\t")
    df = pd.concat([df_train, df_eval], ignore_index=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "id": f"CO_{row['passage_id']}",
                "source": "CO_WIKI",
                "sentence": row["text"],
                "acceptability": 1 if row["label"] == "Orig" else 0,
                "metadata": row["label"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()