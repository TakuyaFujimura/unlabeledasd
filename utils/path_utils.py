# Copyright 2024 Takuya Fujimura

import glob
from pathlib import Path


def get_path_glob(glob_cond):
    path_list = glob.glob(glob_cond)
    if len(path_list) != 1:
        raise FileNotFoundError(f"Error: {len(path_list)} files found for {glob_cond}")
    return path_list[0]


def get_full_path(tmp_path, all_mode=False):
    if "__best__" in tmp_path:
        ret_path = find_best_epoch(tmp_path, all_mode)
    else:
        ret_path = str(get_path_glob(tmp_path))

    return ret_path


def find_best_epoch(tmp_path, all_mode):
    split_path = tmp_path.split("/")
    if split_path[-2] == "checkpoints":
        if all_mode:
            tmp_path = "/".join(split_path[:-3] + ["all"] + split_path[-2:])
        path_dir = Path(tmp_path).parent
    elif split_path[-3] == "infer":
        if all_mode:
            tmp_path = "/".join(split_path[:-4] + ["all"] + split_path[-3:])
        path_dir = Path(tmp_path).parents[2] / "checkpoints"
    else:
        raise NotImplementedError()

    min_loss = float("inf")
    min_path = None
    min_epoch = -1
    for ckpt_path in path_dir.glob("epoch=*.ckpt"):
        loss = float(ckpt_path.stem.split("=")[-1])
        epoch = int(ckpt_path.stem.split("=")[1].split("-")[0])
        if min_loss > loss:
            min_loss = loss
            min_path = ckpt_path
            min_epoch = epoch
        elif min_loss == loss and min_epoch < epoch:
            min_loss = loss
            min_path = ckpt_path
            min_epoch = epoch
    score_path = tmp_path.replace("__best__", min_path.stem)
    if all_mode:
        if split_path[-2] == "checkpoints":
            score_path = score_path.replace("/all/", f"/{split_path[-3]}/")
        elif split_path[-3] == "infer":
            score_path = score_path.replace("/all/", f"/{split_path[-4]}/")

    return score_path
