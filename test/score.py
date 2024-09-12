# Copyright 2024 Takuya Fujimura

import copy
import logging

import pandas as pd
from hydra.utils import instantiate


def zscore(pred):
    return (pred - pred.mean()) / pred.std()


def rm_unnecesary_col(df):
    rm_cols = []
    rm_cols += [col for col in df.keys() if col[0] == "e" and int(col[1:]) >= 0]
    return df.drop(rm_cols, axis=1)


def score_main(config, basename, csv_dir):
    org_df_dict = {}
    output_df_dict = {}
    for split in ["train", "test"]:
        org_df_dict[split] = pd.read_csv(csv_dir / f"{basename}_{split}.csv")
        output_df_dict[split] = copy.deepcopy(org_df_dict[split])
        output_df_dict[split] = rm_unnecesary_col(output_df_dict[split])

    # Loop for fit_data
    for backend_cfg in config.backend:
        backend = instantiate(backend_cfg)
        backend.fit(org_df_dict["train"])
        backend_name = "-".join(
            [backend_cfg._target_.replace("backends.", ""), str(backend_cfg.hp)]
        )
        for split in ["train", "test"]:
            anomaly_score_dict = backend.anomaly_score(org_df_dict[split])
            for key_, score in anomaly_score_dict.items():
                output_df_dict[split][f"AS-{backend_name}-{key_}"] = score

    # Save
    for split, output_df in output_df_dict.items():
        output_path = csv_dir / f"{basename}_{split}_score.csv"
        output_df.to_csv(output_path, index=False)
        logging.info(f"Saved at {str(output_path)}")
