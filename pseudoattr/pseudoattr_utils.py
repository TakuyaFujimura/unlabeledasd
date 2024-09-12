from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def extract(model, loader, device):
    emb_array = np.empty((0, model.embedding_size))
    path_array = np.empty((0))
    for wave, path in tqdm(loader):
        with torch.no_grad():
            embed = model(wave.to(device))["embedding"]
        if isinstance(embed, torch.Tensor):
            embed = embed.cpu().numpy()
        emb_array = np.concatenate([emb_array, embed])
        path_array = np.concatenate([path_array, np.array(path)])
    ecols = [f"e{i}" for i in range(model.embedding_size)]
    df_base = pd.DataFrame.from_dict({"path": path_array})
    df_emb = pd.DataFrame(columns=ecols, data=emb_array)
    df = pd.concat([df_base, df_emb], axis=1)
    return df



def get_train_dom_csv(path, dom):
    df = pd.read_csv(path)
    train_idx = np.array([p.split("/")[-2] == "train" for p in df.path.values])
    if dom is None:
        return df.loc[train_idx]
    else:
        dom_idx = np.array(
            [p.split("/")[-1].split("_")[2] == dom for p in df.path.values]
        )
        return df.loc[train_idx & dom_idx]


def extract_emb(df, keyword="e"):
    ecols = [col for col in df.keys() if col[0] == keyword and int(col[1:]) >= 0]
    return df[ecols].values


def get_org_attr_machines(data_dir, pseudo_attr_machines):
    all_machines = [p.stem for p in Path(f"{data_dir}").glob("*")]
    org_attr_machines = []
    for m in all_machines:
        if m not in pseudo_attr_machines:
            org_attr_machines += [m]
    return org_attr_machines
