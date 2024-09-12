# Copyright 2024 Takuya Fujimura

import glob
import logging
import sys

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from models import STFT
from utils import read_json


def get_basic_label(path, label):
    # section_00_target_train_normal_0009_<attribute>.wav
    split_path_list = path.split("/")[-1].split("_")

    if label == "machine":
        return path.split("/")[-3]
    elif label == "section":
        return int(split_path_list[1])
    elif label == "is_target":
        assert split_path_list[2] in ["source", "target"]
        return int(split_path_list[2] == "target")
    elif label == "is_normal":
        assert split_path_list[4] in ["normal", "anomaly"]
        return int(split_path_list[4] == "normal")
    elif label == "attr":
        return "_".join(split_path_list[6:])
    elif label == "exist_attr":
        return "noAttribute" not in split_path_list[6]
    else:
        raise NotImplementedError()


def can_get_label(path, label):
    if label in ["machine", "section"]:
        return True
    elif label in ["is_target", "is_normal", "attr", "exist_attr"]:
        split_path_list = path.split("/")[-1].split("_")
        return len(split_path_list) > 3
    else:
        return False


class ASDDataset(Dataset):
    def __init__(self, glob_cond_list=[], allow_cache=False, path_list_json=None):
        super().__init__()
        if path_list_json is not None:
            assert len(glob_cond_list) == 0
            self.path_list = read_json(path_list_json)
            self.use_label = False
        else:
            self.path_list = []
            assert len(np.unique(glob_cond_list)) == len(glob_cond_list)
            for glob_cond in glob_cond_list:
                path_list_ = glob.glob(glob_cond)
                if len(path_list_) == 0:
                    raise ValueError(f"No file was found: {glob_cond}")
                self.path_list += path_list_
                logging.info(f"len({glob_cond}) was {len(path_list_)}")
            self.use_label = True
        self.allow_cache = allow_cache
        self.caches_size = len(self.path_list)
        if self.allow_cache:
            self.caches = [None for _ in range(self.caches_size)]

    def __getitem__(self, idx):
        """Get specified idx labels.

        Args:
            idx (int): Index of the item.

        Returns:
            labels: Dict
                labels: (dict).
                wave: (ndarray) Wave (T, ).
        """
        if self.allow_cache and (idx < self.caches_size):
            if self.caches[idx] is not None:
                return self.caches[idx]

        path = self.path_list[idx]
        wave, fs = torchaudio.load(path)
        assert fs == 16000 and wave.shape[0] == 1
        items = {"wave": wave[0]}
        labels = {"path": path}
        if self.use_label:
            for key_ in [
                "machine",
                "section",
                "attr",
                "is_normal",
                "is_target",
                "exist_attr",
            ]:
                if can_get_label(path, key_):
                    labels[key_] = get_basic_label(path, key_)
        items["labels"] = labels

        if self.allow_cache and (idx < self.caches_size):
            self.caches[idx] = items
        return items

    def __len__(self):
        """Return dataset length."""
        return len(self.path_list)


def consecutive_log_spec(melspec, n_frames, stft_power):
    """Make consecutive input feature
    Args:
        melspec: [B, Mel, Time]
    Returns:
        x: [B*n_vectors, dim]
        where n_vectors is number of features obtained from a melspec,
        dim =  n_frames * n_mels , i.e., n_frames-consecutive mel spectrum.
        content of x is like below,
        ---------------------------------
                |n_mels|
                -------------------------
        batch_0 |oooooo|oooooo|oooooo|
                |...                 | n_vectors
                |oooooo|oooooo|oooooo|
                -------------------------
        batch_1 |oooooo|oooooo|oooooo|
                |...                 |
                |oooooo|oooooo|oooooo|
    """
    batch_size = melspec.shape[0]
    log_melspec = (
        20.0
        / stft_power
        * torch.log10(
            torch.maximum(
                melspec, torch.tensor([sys.float_info.epsilon]).to(melspec.device)
            )
        )
    )
    n_vectors = log_melspec.shape[-1] - n_frames + 1
    n_mels = log_melspec.shape[1]
    assert n_vectors > 0
    x = torch.zeros((n_vectors * batch_size, n_mels * n_frames)).to(melspec.device)
    for i in range(batch_size):
        for t in range(n_frames):
            x[
                n_vectors * i : n_vectors * (i + 1),
                n_mels * t : n_mels * (t + 1),
            ] = log_melspec[i, :, t : t + n_vectors].T
    return x


class ConsecutiveSpecDataset(Dataset):
    def __init__(
        self,
        glob_cond_list: list,
        stft_cfg: dict,
        n_frames: int,
        use_machine: bool = False,
    ):
        super().__init__()
        path_list = []
        self.stft = STFT(**stft_cfg)
        self.n_frames = n_frames
        self.use_machine = use_machine
        assert len(np.unique(glob_cond_list)) == len(glob_cond_list)
        for glob_cond in glob_cond_list:
            path_list_ = glob.glob(glob_cond)
            assert len(path_list_) > 0
            path_list += path_list_
            logging.info(f"len({glob_cond}) was {len(path_list_)}")

        feat_list = []
        self.machine_list = []
        for path in path_list:
            wave, fs = torchaudio.load(path)
            assert fs == 16000 and wave.shape[0] == 1
            melspec = self.stft(wave)
            x = consecutive_log_spec(melspec, self.n_frames, self.stft.power)
            feat_list += [x]  # (B', F')
            self.machine_list += [path.split("/")[-3] for _ in range(len(x))]
        self.feat_list = torch.concat(feat_list, dim=0)

    def __getitem__(self, idx):
        if self.use_machine:
            return self.feat_list[idx], self.machine_list[idx]
        else:
            return self.feat_list[idx]

    def __len__(self):
        """Return dataset length."""
        return len(self.feat_list)
