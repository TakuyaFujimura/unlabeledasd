# Copyright 2024 Takuya Fujimura

from typing import Dict

import copy
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import hmean
from torch import Tensor, nn

from hydra.utils import instantiate

from .base_model import BasePLModel
from .ssl_model import get_idx
from models import mix_rand_snr


class AEPLModel(BasePLModel):
    def __init__(self, config):
        super().__init__(config)

    def _constructor(self, module_cfg={}) -> None:
        self.model = instantiate(module_cfg)
        self.embedding_size = self.model.embedding_size
        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(self, wave: Tensor) -> Dict:
        """
        Args:
            x (Tensor): wave (B, T)
        """
        input_feat = self.model.extract_feat(wave)
        # (B*n_vectors, F*n_frames)

        output_dict = self.model(input_feat)
        loss = self.loss_fn(output_dict["reconstructed"], input_feat)
        output_dict["anomaly_score"] = {
            "plain": torch.mean(loss.reshape(len(wave), -1), dim=-1)
        }
        output_dict["embedding"] = output_dict["embedding"].reshape(len(wave), -1)
        return output_dict

    def training_step(self, x, batch_idx):
        output_dict = self.model(x)
        loss = self.loss_fn(output_dict["reconstructed"], x)
        loss_dict = {"main": torch.mean(loss)}

        self.log_loss(torch.tensor(len(x)).float(), f"train/batch_size", 1)
        for tag_, val_ in loss_dict.items():
            self.log_loss(val_, f"train/{tag_}", len(x))

        return loss_dict["main"]

    def on_validation_start(self):
        self.valid_is_target = np.empty(0)
        self.valid_is_normal = np.empty(0)
        self.valid_anom_score = {}

    def validation_step(self, batch, batch_idx):
        output_dict = self(batch["wave"])
        for score_name, score in output_dict["anomaly_score"].items():
            if score_name in self.valid_anom_score:
                self.valid_anom_score[score_name] = np.concatenate(
                    [self.valid_anom_score[score_name], score.cpu().numpy()]
                )
            else:
                self.valid_anom_score[score_name] = score.cpu().numpy()
        self.valid_is_normal = np.concatenate(
            [self.valid_is_normal, batch["is_normal"]]
        )
        self.valid_is_target = np.concatenate(
            [self.valid_is_target, batch["is_target"]]
        )

    def on_validation_end(self):
        auc_idx_dict = {
            "AUC_source_only": self.valid_is_target == 0,
            "AUC_target_only": self.valid_is_target == 1,
            "AUC_source_all": (self.valid_is_target == 0) | (self.valid_is_normal == 0),
            "AUC_target_all": (self.valid_is_target == 1) | (self.valid_is_normal == 0),
        }
        for score_name, score in self.valid_anom_score.items():
            auc_result = {}
            for auc_name, tgt_idx in auc_idx_dict.items():
                auc_result[auc_name] = roc_auc_score(
                    1 - self.valid_is_normal[tgt_idx], score[tgt_idx]
                )
            auc_result["AUC_pauc_all"] = roc_auc_score(
                1 - self.valid_is_normal, score, max_fpr=0.1
            )
            for auc_name, auc in auc_result.items():
                self.logger.log_metrics(
                    {f"valid/{auc_name}/{score_name}": auc},
                    step=self.trainer.global_step,
                )
            official = hmean(
                [
                    auc_result["AUC_source_all"],
                    auc_result["AUC_target_all"],
                    auc_result["AUC_pauc_all"],
                ]
            )
            self.logger.log_metrics(
                {f"valid/official/{score_name}": official},
                step=self.trainer.global_step,
            )
