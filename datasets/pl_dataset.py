# Copyright 2024 Takuya Fujimura

import logging

import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from utils import read_json, write_pkl

from .collators import ASDCollator
from .labelencoder import Labeler
from .torch_dataset import ASDDataset


class ASDDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.datasets = {}

    def setup(self, stage):
        for split in ["train"]:
            dataset = ASDDataset(**self.cfg.datamodule[split].dataset)
            if self.cfg.datamodule[split].batch_sampler is None:
                batch_sampler = None
            else:
                batch_sampler = instantiate(
                    {"dataset": dataset, **self.cfg.datamodule[split].batch_sampler}
                )
            self.datasets[split] = [dataset, batch_sampler]
        self.create_le_dict()
        self.write_info()

    def create_le_dict(self):
        self.le_dict = {}
        if not hasattr(self.cfg.model.model_cfg, "label_dict"):
            logging.warning("self.cfg.model does not have label_dict")
            return

        for key_ in self.cfg.model.model_cfg.label_dict:
            if key_ in self.cfg.datamodule.pretrained_single_idx_dict_path_dict:
                pretrained_dict = read_json(
                    self.cfg.datamodule.pretrained_single_idx_dict_path_dict[key_]
                )
                self.le_dict[key_] = Labeler(key_, pretrained_dict, "single")
            elif key_ in self.cfg.datamodule.pretrained_multi_idx_dict_path_dict:
                pretrained_dict = read_json(
                    self.cfg.datamodule.pretrained_multi_idx_dict_path_dict[key_]
                )
                self.le_dict[key_] = Labeler(key_, pretrained_dict, "multi")
            else:
                self.le_dict[key_] = Labeler(key_)

            self.le_dict[key_].fit(self.datasets["train"][0].path_list)
        ## Save le_dict
        write_pkl(self.cfg.datamodule.le_path, self.le_dict)

    def write_info(self):
        if getattr(self.cfg.model, "scheduler_cfg") is not None:
            raise NotImplementedError()
        ## Write n_head_dict
        if hasattr(self.cfg.model.model_cfg, "label_dict"):
            for key_ in self.cfg.model.model_cfg.label_dict:
                with open_dict(self.cfg.model.model_cfg.label_dict):
                    self.cfg.model.model_cfg.label_dict[key_]["num"] = self.le_dict[
                        key_
                    ].num

    def _get_loader(self, split):
        # Set
        collator = ASDCollator(**self.cfg.datamodule[split].collator)
        return DataLoader(
            dataset=self.datasets[split][0],
            batch_sampler=self.datasets[split][1],
            collate_fn=collator,
            **self.cfg.datamodule[split].dataloader,
        )

    def train_dataloader(self):
        return self._get_loader("train")

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None


class ASDGenDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.datasets = {}

    def setup(self, stage):
        for split in ["train", "valid"]:
            dataset = instantiate(self.cfg.datamodule[split].dataset)
            if self.cfg.datamodule[split].batch_sampler is None:
                batch_sampler = None
            else:
                batch_sampler = instantiate(
                    {"dataset": dataset, **self.cfg.datamodule[split].batch_sampler}
                )
            if self.cfg.datamodule[split].collator is None:
                collator = None
            else:
                collator = ASDCollator(**self.cfg.datamodule[split].collator)
            self.datasets[split] = [dataset, batch_sampler, collator]

    def _get_loader(self, split):
        return DataLoader(
            dataset=self.datasets[split][0],
            batch_sampler=self.datasets[split][1],
            collate_fn=self.datasets[split][2],
            **self.cfg.datamodule[split].dataloader,
        )

    def train_dataloader(self):
        return self._get_loader("train")

    def val_dataloader(self):
        return self._get_loader("valid")

    def test_dataloader(self):
        return None
