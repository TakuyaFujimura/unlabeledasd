# Copyright 2024 Takuya Fujimura

import logging
from pathlib import Path

import numpy as np
import torch

from utils import read_pkl


def to_torch_from_list(list_items, key_list, wave_key):
    items = {}
    items[wave_key] = torch.stack(list_items[wave_key])
    for key_ in key_list:
        if (
            isinstance(list_items[key_][0], str)
            or isinstance(list_items[key_][0], bool)
            or isinstance(list_items[key_][0], int)
            or isinstance(list_items[key_][0], float)
        ):
            items[key_] = np.array(list_items[key_])
        else:
            raise NotImplementedError()
    return items


class ASDCollator(object):
    """Wave form data's collator."""

    def __init__(
        self,
        sr=16000,
        sec=3,
        le_path="",
        shuffle=True,
        fixed_start_frame=0,
        pad_type="tile",
        use_le=True,  # To controll the behavior in the test
    ):
        """Initialize customized collator for PyTorch DataLoader."""
        self.sr = sr
        self.sec = sec
        if Path(le_path).exists():
            logging.info(f"Load labelencoder form {le_path}")
            self.le_dict = read_pkl(le_path)
        else:
            logging.warning(f"{le_path} is not available")
            self.le_dict = {}
        self.shuffle = shuffle
        self.fixed_start_frame = fixed_start_frame
        if isinstance(sec, int):
            self.crop_len = int(sr * sec)
        elif sec == "all":
            self.crop_len = "all"
        else:
            raise NotImplementedError()
        self.pad_type = pad_type
        self.use_le = use_le

    def get_start_rand_frame(self, wave):
        return torch.randint(0, max(1, len(wave) - self.crop_len), (1,))[0]

    def wave_crop(self, wave):
        if self.crop_len == "all":
            return wave
        if len(wave) < self.crop_len:
            if self.pad_type == "tile":
                wave = wave.tile(int(np.ceil(self.crop_len / len(wave))))
                wave = wave[: self.crop_len]
            elif self.pad_type == "rand_tile":
                wave = wave.tile(int(np.ceil(self.crop_len / len(wave))))
            else:
                raise NotImplementedError()
        start_frame = (
            self.get_start_rand_frame(wave) if self.shuffle else self.fixed_start_frame
        )
        return wave[start_frame : start_frame + self.crop_len]

    def stack_batch(self, batch):
        key_list = batch[0]["labels"].keys()
        list_items = {key_: [] for key_ in key_list}
        list_items["wave"] = []

        for b in batch:
            assert key_list == b["labels"].keys()
            list_items["wave"].append(self.wave_crop(b["wave"]))
            for key_ in key_list:
                list_items[key_].append(b["labels"][key_])

        return list_items, key_list

    def add_le_labels(self, items):
        for key_, le in self.le_dict.items():
            # key_ is "onehot_hogehoge"
            ret_dict = le.trans(items["path"])
            items[f"{key_}_idx"] = ret_dict["idx"]
            items[key_] = ret_dict["onehot"]
        return items

    def __call__(self, batch):
        """Convert into batch tensors."""
        # list of dict -> dict of list
        list_items, key_list = self.stack_batch(batch)
        items = to_torch_from_list(list_items, key_list, "wave")
        if self.use_le:
            items = self.add_le_labels(items)
        return items
