import seaborn as sns
import pandas as pd
import numpy as np
import h5py
import re

sns.set_theme(style="whitegrid", font_scale=0.9)


def sort_model_names(model_names):
    def sort_key(name):
        layers = int(re.search(r'(\d+)l', name).group(1))
        heads  = int(re.search(r'(\d+)h', name).group(1))
        return (layers, heads)
    
    return sorted(model_names, key=sort_key)

def load_representations(representations_path, model_name, dataset_name):
    with h5py.File(representations_path, "r") as f:
        data = f[model_name][dataset_name]
        ids  = data["ids"].asstr()[:] 
        embeddings = data["last_token"][:]    # (N, num_layers, H)

    return ids, embeddings

def load_acceptability_mask(dataset_path, ids):
    dataset = pd.read_json(dataset_path, lines=True).set_index("id")
    acceptability = dataset.loc[ids]['acceptability'].tolist()
    acceptability_mask = np.array(acceptability, dtype=bool)
    return acceptability_mask
