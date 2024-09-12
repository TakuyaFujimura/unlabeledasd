# Copyright 2024 Takuya Fujimura

from typing import Dict

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch import Tensor

from models import CutMixLayer, FeatExLayer, MixupLayer

from .base_model import BasePLModel


class FeatExPLModel(BasePLModel):
    def __init__(self, config):
        super().__init__(config)

    def _constructor(
        self,
        label_dict={},
        extractor_cfg={},
        loss_cfg={},
        mixup_prob=0.5,
        featex_prob=0.5,
        mixup_type="mixup",
        lam=1.0,
    ) -> None:
        self.label_dict = label_dict
        self.extractor = instantiate(extractor_cfg)
        self.loss_cfg = loss_cfg
        self.embedding_size = self.extractor.embedding_size
        self.embedding_split_size = self.extractor.embedding_split_size
        self.embedding_split_num = self.extractor.embedding_split_num
        self.featex_prob = featex_prob
        self.featex = FeatExLayer(
            featex_prob,
            list(self.label_dict.keys()),
            self.embedding_split_num,
        )
        self.lam = lam

        if mixup_type == "cutmix":
            self.mixup = CutMixLayer(
                mixup_prob,
                list(self.label_dict.keys()),
            )
        elif mixup_type == "mixup":
            self.mixup = MixupLayer(mixup_prob, list(self.label_dict.keys()))
        else:
            raise NotImplementedError()

        self.head_dict = torch.nn.ModuleDict({})
        for key_, dict_ in self.label_dict.items():
            self.head_dict[key_] = torch.nn.ModuleDict({})
            for loss_name, i in zip(
                ["mixup", "featex"], [1, (self.embedding_split_num + 1)]
            ):
                loss_cfg = {
                    **{
                        "n_classes": dict_["num"] * i,
                        "embed_size": self.embedding_size,
                        "trainable": loss_name == "featex",
                    },
                    **self.loss_cfg,
                }
                self.head_dict[key_][loss_name] = instantiate(loss_cfg)

    def forward(self, wave: Tensor) -> Dict:
        """
        Args:
            x (Tensor): wave (B, T)
        """
        embedding, _ = self.extractor(wave)
        return {"embedding": F.normalize(embedding, dim=1)}

    def training_step(self, batch, batch_idx):
        wave = batch.pop("wave")
        labels = batch  # just renamed

        wave, labels_mixup = self.mixup(wave, labels)
        embedding, z_list = self.extractor(wave)
        embedding_featex, labels_featex = self.featex(z_list, labels_mixup)

        loss_dict = {"main": 0.0}
        for key_ in self.label_dict:
            l_mixup = self.head_dict[key_]["mixup"](embedding, labels_mixup[key_])

            if self.featex_prob > 0.0:
                l_featex = self.head_dict[key_]["featex"](
                    embedding_featex, labels_featex[key_]
                )
            else:
                l_featex = 0.0

            loss_dict[key_] = l_mixup + self.lam * l_featex
            loss_dict["main"] += loss_dict[key_] * self.label_dict[key_]["lam"]

        self.log_loss(torch.tensor(len(wave)).float(), f"train/batch_size", 1)
        for tag_, val_ in loss_dict.items():
            self.log_loss(val_, f"train/{tag_}", len(wave))

        return loss_dict["main"]
