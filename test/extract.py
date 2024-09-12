# Copyright 2024 Takuya Fujimura

import logging

import numpy as np
import pandas as pd
import torch
from relpath import add_import_path
from torch.utils.data import DataLoader

add_import_path("..")
from datasets import ASDCollator, ASDDataset


def make_empty_df_data(plmodel=None, is_eval=False):
    if is_eval:
        key_list = ["path"]
    else:
        key_list = ["path", "is_normal", "is_target"]

    if (plmodel is not None) and hasattr(plmodel, "label_dict"):
        key_list += [f"{l}_idx" for l in plmodel.label_dict.keys()]
    empty_df_data = {}
    for key_ in key_list:
        empty_df_data[key_] = np.empty((0))
    return empty_df_data


def make_df(df_data, embed, anomaly_score):
    # remove len-0 element
    rm_key_list = []
    for key_ in df_data:
        if len(df_data[key_]) == 0:
            rm_key_list += [key_]
    for key_ in rm_key_list:
        df_data.pop(key_)
    # make DataFrame
    df_base = pd.DataFrame.from_dict(df_data)
    df_as = pd.DataFrame.from_dict(
        {f"AS-{name}": score for name, score in anomaly_score.items()}
    )
    embed_cols = [f"e{i}" for i in range(embed.shape[-1])]
    df_emb = pd.DataFrame(columns=embed_cols, data=embed)
    df = pd.concat([df_base, df_as, df_emb], axis=1)
    return df


def make_loader_dict(config, past_cfg):
    if hasattr(past_cfg.datamodule.train.collator, "shuffle"):
        past_cfg.datamodule.train.collator.shuffle = False
        logging.info("Shuffle of Collator is set to False")
    loader_dict = {}
    for split in ["train", "test"]:
        if hasattr(past_cfg.datamodule, "valid"):
            assert past_cfg.datamodule.gen
            collator_cfg = past_cfg.datamodule.valid.collator
        else:
            collator_cfg = past_cfg.datamodule.train.collator
        collator = ASDCollator(**collator_cfg, use_le=split == "train")
        glob_cond = f"{past_cfg.datamodule.data_dir}/{config.pos_machine}/{split}/*.wav"
        dataset = ASDDataset(glob_cond_list=[glob_cond])
        loader_dict[split] = DataLoader(
            dataset=dataset,
            collate_fn=collator,
            shuffle=False,
            **config.dataloader_cfg,
        )
    return loader_dict


def extract(dataloader, plmodel, device, is_eval=False):
    """
    Args:
        dataloader: dataloader
        device: device
    Returns:
        df: df including audio_info and embedding
    """
    logging.info("Start extract_loader")

    plmodel.eval()
    embed = None  # np.empty((0, plmodel.embedding_size))
    anomaly_score = {}
    df_data = make_empty_df_data(None, is_eval)
    # df_data = make_empty_df_data(plmodel, is_eval) # when label_idx is needed

    for batch in dataloader:
        with torch.no_grad():
            wave = batch["wave"].to(device)
            output_dict = plmodel(wave)
            if "embedding" in output_dict:
                # Store embedding
                if embed is None:
                    embed = output_dict["embedding"].cpu().numpy()
                else:
                    embed = np.concatenate(
                        [embed, output_dict["embedding"].cpu().numpy()]
                    )
            if "anomaly_score" in output_dict:
                # Store anomaly scores
                for key, scores in output_dict["anomaly_score"].items():
                    if key in anomaly_score:
                        anomaly_score[key] = np.concatenate(
                            [anomaly_score[key], scores.cpu().numpy()]
                        )
                    else:
                        anomaly_score[key] = scores.cpu().numpy()

        for key_ in df_data:
            if key_ in batch:
                df_data[key_] = np.concatenate([df_data[key_], batch[key_]])
            elif key_ in output_dict:
                df_data[key_] = np.concatenate([df_data[key_], output_dict[key_]])
            else:
                assert key_[-4:] == "_idx"
    df = make_df(df_data, embed, anomaly_score)
    return df


def extract_main(config, plmodel, basename, csv_dir, past_cfg):
    loader_dict = make_loader_dict(config, past_cfg)
    for split, loader in loader_dict.items():
        is_eval = getattr(config, "is_eval", False) and (split == "test")
        df = extract(loader, plmodel, config.device, is_eval)
        csv_path = csv_dir / f"{basename}_{split}.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved extracted data at {csv_path}")
