import pandas as pd
import json
import os


def write_df_to_file(df, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "id": f"PT_{row['ID']}",
                "source": "PT",
                "sentence": row["Sentence"],
                "acceptability": int(row["Labels"]),
                "metadata": row["Construction"] if "Construction" in df.columns else "NA"
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
def main():
    datasets_dir = "data/acceptance_datasets/src"
    train_paths = [os.path.join(datasets_dir, file_name) for file_name in os.listdir(datasets_dir) if "pretens-It-Subtask1-fold_" in file_name]
    test_path = "data/acceptance_datasets/src/pretens-It-Subtask1-labels.tsv"
    out_dir = "data/acceptance_datasets/"

    train_dfs = [pd.read_csv(train_path, sep="\t") for train_path in train_paths]
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df["ID"] = train_df["ID"].astype(str) + "_train"
    train_path = os.path.join(out_dir, "train", "pretens_it.jsonl")
    write_df_to_file(train_df, train_path)

    test_df = pd.read_csv(test_path, sep="\t")
    test_df["ID"] = test_df["ID"].astype(str) + "_test"
    test_path = os.path.join(out_dir, "test", "pretens_it.jsonl")
    write_df_to_file(test_df, test_path)

    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    merged_path = os.path.join(out_dir, "clean_merged", "pretens_it.jsonl")
    write_df_to_file(merged_df, merged_path)

if __name__ == "__main__":
    main()