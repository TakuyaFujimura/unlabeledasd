# Copyright 2024 Takuya Fujimura

import random
import logging

from torch.utils.data.sampler import BatchSampler

from .torch_dataset import get_basic_label


class PosNegSampler(BatchSampler):
    """BatchSampler - positive:negative = 1 : 1.

    Returns batches of size n_classes * n_samples
    """

    def __init__(
        self,
        dataset,
        pos_machine,
        batch_size=64,
        shuffle=False,
        drop_last=False,
    ):
        """Batch Sampler.

        Args:
            dataset (dataset): dataset for ASD
            batch_size (int, optional): batch size. Defaults to 64.
            shuffle (bool, optional): shuffle. Defaults to False.
            drop_last (bool, optional): drop last. Defaults to False.
            n_target (int, optional): The number of target sample. Defaults to 1.
        """
        path_list = dataset.path_list
        self.pos_machine = pos_machine
        logging.info(f"pos_machine is {self.pos_machine}")
        self.set_idx(path_list)
        self.used_idx_cnt = {"pos": 0, "neg": 0}
        self.batch_size = batch_size
        assert batch_size % 2 == 0
        self.n_samples = batch_size // 2  #  positive:negative = 1 : 1
        self.dataset = dataset
        self.shuffle = shuffle
        self.drop_last = drop_last

    def set_idx(self, path_list):
        self.idxs_dict = {"pos": [], "neg": []}
        for i, aud_path in enumerate(path_list):
            if get_basic_label(aud_path, "machine") == self.pos_machine:
                self.idxs_dict["pos"].append(i)
            else:
                self.idxs_dict["neg"].append(i)
        self.n_pos = len(self.idxs_dict["pos"])
        self.n_neg = len(self.idxs_dict["neg"])
        logging.info(f"number of pos: {self.n_pos}")
        logging.info(f"number of neg: {self.n_neg}")
        assert self.n_pos > 0
        assert self.n_pos + self.n_neg == len(path_list)

    def __iter__(self):
        self.used_idx_cnt["pos"] = 0
        self.used_idx_cnt["neg"] = 0

        if self.shuffle:
            random.shuffle(self.idxs_dict["pos"])
            random.shuffle(self.idxs_dict["neg"])
        while self.used_idx_cnt["pos"] + self.n_samples <= self.n_pos:
            indices = []
            for split in ["pos", "neg"]:
                indices.extend(
                    self.idxs_dict[split][
                        self.used_idx_cnt[split] : self.used_idx_cnt[split]
                        + self.n_samples
                    ]
                )
                self.used_idx_cnt[split] += self.n_samples
            if self.shuffle:
                random.shuffle(indices)
            yield indices

        if not self.drop_last and self.n_pos - self.used_idx_cnt["pos"] > 0:
            indices = []
            indices.extend(self.idxs_dict["pos"][self.used_idx_cnt["pos"] :])
            indices.extend(
                self.idxs_dict["neg"][
                    self.used_idx_cnt["neg"] : self.used_idx_cnt["neg"] + len(indices)
                ]
            )
            yield indices

    def __len__(self):
        if self.drop_last or (self.n_pos % self.n_samples == 0):
            return self.n_pos // self.n_samples
        else:
            return (self.n_pos // self.n_samples) + 1


class ConsecutiveSpecPosNegSampler(PosNegSampler):
    """BatchSampler - positive:negative = 1 : 1.

    Returns batches of size n_classes * n_samples
    """

    def __init__(
        self,
        dataset,
        pos_machine,
        batch_size=64,
        shuffle=False,
        drop_last=False,
    ):
        machine_list = dataset.machine_list
        self.pos_machine = pos_machine
        logging.info(f"pos_machine is {self.pos_machine}")
        self.set_idx(machine_list)
        self.used_idx_cnt = {"pos": 0, "neg": 0}
        self.batch_size = batch_size
        assert batch_size % 2 == 0
        self.n_samples = batch_size // 2  #  positive:negative = 1 : 1
        self.dataset = dataset
        self.shuffle = shuffle
        self.drop_last = drop_last

    def set_idx(self, machine_list):
        self.idxs_dict = {"pos": [], "neg": []}
        for i, machine in enumerate(machine_list):
            if machine == self.pos_machine:
                self.idxs_dict["pos"].append(i)
            else:
                self.idxs_dict["neg"].append(i)
        self.n_pos = len(self.idxs_dict["pos"])
        self.n_neg = len(self.idxs_dict["neg"])
        logging.info(f"number of pos: {self.n_pos}")
        logging.info(f"number of neg: {self.n_neg}")
        assert self.n_pos > 0
        assert self.n_pos + self.n_neg == len(machine_list)
