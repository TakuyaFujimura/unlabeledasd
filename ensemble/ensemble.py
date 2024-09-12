# Copyright 2024 Takuya Fujimura

import argparse
import datetime
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from relpath import rel2abs
from tqdm import tqdm


def get_path(glob_cond):
    path_list = glob.glob(glob_cond)
    if len(path_list) != 1:
        raise ValueError(f"Files not found or too many files found: {glob_cond}")
    return path_list[0]


def adjust_lam(cfg_list):
    lam_list = []
    # convert lam to probability
    for cfg in cfg_list:
        lam_list += [getattr(cfg, "lam", 1.0)]
    lam_list = np.array(lam_list) / sum(lam_list)
    # replace it
    for i, cfg in enumerate(cfg_list):
        cfg["lam"] = lam_list[i].item()


def save_info(infer_dir):
    OmegaConf.save(config, infer_dir / "ensemble_setting.yaml")
    with open(infer_dir / "info.txt", "w") as f:
        f.write(f"{datetime.datetime.now()}\n")


def main(config, machines):
    adjust_lam(config.ensemble_list)

    for m in tqdm(machines):
        infer_dir = Path(f"{config.save_dir}/{m}/infer/version_{config.infer_version}")
        infer_dir.mkdir(parents=True, exist_ok=False)
        save_info(infer_dir)
        # for split in ["test"]:
        for split in ["train", "test"]:
            ensemble_df_data = {
                "path": None,
                "AS-ensemble": None,
            }
            for i, cfg in enumerate(config.ensemble_list):
                score_path = get_path(
                    f"{cfg.result_dir}/{m}/infer/version_{cfg.infer_version}/*_{split}_score.csv"
                )
                df = pd.read_csv(score_path).sort_values(by="path")

                if ensemble_df_data["path"] is None:
                    for key in ["path", "is_normal", "is_target"]:
                        if key in df.columns:
                            ensemble_df_data[key] = df[key].values
                    ensemble_df_data["AS-ensemble"] = np.zeros(len(df))
                else:
                    for key in ["path", "is_normal", "is_target"]:
                        if key in df.columns:
                            assert np.all(ensemble_df_data[key] == df[key].values)

                ensemble_df_data["AS-ensemble"] += cfg.lam * df[cfg.backend].values
            output_df = pd.DataFrame(ensemble_df_data)
            output_df.to_csv(infer_dir / f"ensemble_{split}_score.csv", index=False)


def replace_base_dir(config, base_dir=None, seed=None):
    if base_dir is not None:
        config.save_dir = config.save_dir.replace("<base_dir>", base_dir)
        for cfg in config.ensemble_list:
            cfg.result_dir = cfg.result_dir.replace("<base_dir>", base_dir)
    if seed is not None:
        config.save_dir = config.save_dir.replace("<seed>", seed)
        for cfg in config.ensemble_list:
            cfg.result_dir = cfg.result_dir.replace("<seed>", seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--machines", type=str, nargs="+")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--seed", type=str, default=None)
    args = parser.parse_args()
    config = OmegaConf.load(rel2abs(args.config_path))
    replace_base_dir(config, args.base_dir, args.seed)
    main(config, args.machines)
