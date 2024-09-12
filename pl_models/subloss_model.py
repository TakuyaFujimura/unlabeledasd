# Copyright 2024 Takuya Fujimura

from typing import Dict

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch import Tensor

from models import CutMixLayer, MixupLayer, get_perm

from .base_model import BasePLModel


class SubspaceLossPLModel(BasePLModel):
    def __init__(self, config):
        super().__init__(config)

    def _constructor(
        self,
        label_dict={},
        extractor_cfg={},
        loss_cfg={},
        mixup_prob=0.5,
        mixup_type="mixup",
        lam=1.0,
        aug_cfg_list=[],
    ) -> None:
        self.label_dict = label_dict
        self.extractor = instantiate(extractor_cfg)
        self.loss_cfg = loss_cfg
        self.embedding_size = self.extractor.embedding_size
        self.embedding_split_size = self.extractor.embedding_split_size
        self.embedding_split_num = self.extractor.embedding_split_num
        self.mixup_prob = mixup_prob
        self.mixup_type = mixup_type

        if self.mixup_type == "cutmix":
            self.mixup = CutMixLayer(mixup_prob, list(self.label_dict.keys()))
        elif self.mixup_type == "mixup":
            self.mixup = MixupLayer(mixup_prob, list(self.label_dict.keys()))
        else:
            raise NotImplementedError()

        self.main_head_dict = torch.nn.ModuleDict({})
        self.split_head_dict = torch.nn.ModuleDict({})
        for key_, dict_ in self.label_dict.items():
            main_loss_cfg = {
                **{
                    "n_classes": dict_["num"],
                    "embed_size": self.embedding_size,
                    "trainable": False,
                },
                **self.loss_cfg,
            }
            self.main_head_dict[key_] = instantiate(main_loss_cfg)
            split_loss_cfg = {
                **{
                    "n_classes": dict_["num"],
                    "embed_size": self.embedding_split_size,
                    "trainable": True,
                },
                **self.loss_cfg,
            }
            self.split_head_dict[key_] = torch.nn.ModuleList(
                [instantiate(split_loss_cfg) for _ in range(self.embedding_split_num)]
            )
        self.lam = lam

        self.augmentations = torch.nn.ModuleList([])
        for cfg in aug_cfg_list:
            self.augmentations.append(instantiate(cfg))

    def forward(self, wave: Tensor) -> Dict:
        """
        Args:
            x (Tensor): wave (B, T)
        """
        embedding, _ = self.extractor(wave)
        return {"embedding": F.normalize(embedding, dim=1)}

    def training_step(self, batch, batch_idx):
        for aug_func in self.augmentations:
            batch = aug_func(batch)

        wave = batch.pop("wave")
        labels = batch  # just renamed

        wave, labels_mixup = self.mixup(wave, labels)
        embedding, z_list = self.extractor(wave)
        assert len(z_list) == self.embedding_split_num

        loss_dict = {"main": 0.0}
        for key_ in self.label_dict:
            l_main = self.main_head_dict[key_](embedding, labels_mixup[key_])
            l_other = 0.0
            for i, z in enumerate(z_list):
                loss_dict[f"{key_}_other_{i}"] = self.split_head_dict[key_][i](
                    z, labels_mixup[key_]
                )
                l_other += loss_dict[f"{key_}_other_{i}"]
            loss_dict[key_] = l_main + self.lam * l_other
            loss_dict[f"{key_}_main"] = l_main
            loss_dict["main"] += loss_dict[key_] * self.label_dict[key_]["lam"]

        self.log_loss(torch.tensor(len(wave)).float(), "train/batch_size", 1)
        for tag_, val_ in loss_dict.items():
            self.log_loss(val_, f"train/{tag_}", len(wave))

        return loss_dict["main"]

