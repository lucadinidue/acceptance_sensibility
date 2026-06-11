from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import argparse
import torch
import h5py
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def save_embeddings(embeddings, ids, output_path,  model_name, dataset_key):
    with h5py.File(output_path, "a") as out_file:
        grp = out_file.require_group(f"{model_name}/{dataset_key}")
        grp.create_dataset("last_token", data=embeddings, compression="gzip")  # (N, num_layers, H)
        grp.create_dataset("ids", data=np.array(ids, dtype=h5py.string_dtype()))


@torch.inference_mode()
def get_last_token_embeddings(model, texts, tokenizer, device, batch_size:int=8, include_embedding_matrix=True):
    model.eval()
    all_embs = []

    def collate(batch_texts):
        return tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)

    loader = DataLoader(texts, batch_size=batch_size, shuffle=False, collate_fn=collate)
    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)            # (B, L)
        attention_mask = batch["attention_mask"].to(device)  # (B, L)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states if include_embedding_matrix else outputs.hidden_states[1:]
        hidden_states = torch.stack(hidden_states, dim=0)        # (num_layers, B, L, H)

        # Taking the last token embedding for each sequence
        seq_len = attention_mask.size(1)
        last_idx = attention_mask.sum(dim=1) - 1 # (B,)
        batch_arange = torch.arange(input_ids.size(0), device=device)   # (B,)

        # Take last token on every layer
        sent_emb = hidden_states[:, batch_arange, last_idx, :] #  (num_layers, B, H)
        sent_emb = sent_emb.permute(1, 0, 2).contiguous() #  (B, num_layers, H)
        all_embs.append(sent_emb.cpu())

    return torch.cat(all_embs, dim=0)                            # (N, num_layers, H)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Representation Extraction")
    parser.add_argument("--model_name", type=str, help="Path of the model")
    parser.add_argument("--dataset_path", type=str, help="Path of the dataset (jsonl)")
    parser.add_argument("--output_dir", type=str, default="data/representations", help="Directory to save the output HDF5 file")
    args = parser.parse_args()

    if "coherence" in args.dataset_path:
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    data = load_dataset("json", data_files=args.dataset_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, output_hidden_states=True)
    model.eval().to(device)

    embeddings = get_last_token_embeddings(model, data["sentence"], tokenizer, device)
    embeddings = embeddings.numpy() #.astype(np.float32)   # (N, num_layers, H)

    model_key = args.model_name.split("/")[-2]
    dataset_key = args.dataset_path.split('/')[-1].split(".")[0]
    output_path = os.path.join(args.output_dir, f'{dataset_key}_{model_key}.h5')
    save_embeddings(embeddings, data["id"], output_path, model_key, dataset_key)

if __name__ == "__main__":
    main()