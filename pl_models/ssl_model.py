# Copyright 2024 Takuya Fujimura

import copy
import math
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from librosa import amplitude_to_db
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from torchvision.transforms.functional import resize

from models import STFT, STFT2dEncoderLayer, mix_rand_snr, rand_uniform_tensor

from .base_model import BasePLModel


def get_idx(prob, num, device):
    dec = torch.rand(num, device=device) < prob
    return dec


def statex(X, N, t_prob=0.5):
    assert len(X.shape) == 3 and len(N.shape) == 3
    N = N[torch.randperm(len(N))[: len(X)].to(X.device)]

    X_tex = (
        (X - X.mean(dim=2, keepdim=True)) / (X.std(dim=2, keepdim=True) + 1e-6)
    ) * N.std(dim=2, keepdim=True) + N.mean(dim=2, keepdim=True)
    X_fex = (
        (X - X.mean(dim=1, keepdim=True)) / (X.std(dim=1, keepdim=True) + 1e-6)
    ) * N.std(dim=1, keepdim=True) + N.mean(dim=1, keepdim=True)
    dec_t = torch.rand(len(X), device=X.device) < t_prob
    dec_t = dec_t.reshape([-1] + [1] * (len(X.shape) - 1)).float()
    return dec_t * X_tex + (1 - dec_t) * X_fex


def stretch(X, min_ratio_int=[0.5, 0.8], max_ratio_int=[1.2, 1.5], t_prob=0.5):
    """
    Args
        X: spectrogram (B, F, T)
        t_prob:
        min_ratio_int: min_ratio interval
        max_ratio_int: max_ratio interval
    """
    result = torch.zeros_like(X, device=X.device)
    dec_t_batch = torch.rand(len(X), device=X.device) < t_prob
    F, T = X.shape[1:]
    for i, dec_t in enumerate(dec_t_batch):
        if torch.rand(1).item() < 0.5:
            ratio = rand_uniform_tensor([1], min_ratio_int[0], min_ratio_int[1]).item()
        else:
            ratio = rand_uniform_tensor([1], max_ratio_int[0], max_ratio_int[1]).item()

        if dec_t:
            F_modified, T_modified = F, int(T * ratio)
        else:
            F_modified, T_modified = int(F * ratio), T

        Y = resize(X[i][None, None], [F_modified, T_modified], antialias=True)[0, 0]

        if dec_t and ratio < 1.0:
            Y = torch.tile(Y, [1, int(math.ceil(T / T_modified))])
            Y = Y[:, :T]
            T_modified = T
        F_crop, T_crop = min(F, F_modified), min(T, T_modified)
        result[i][:F_crop, :T_crop] += Y[:F_crop, :T_crop]
    return result


class ContrastivePLModel(BasePLModel):
    def __init__(self, config):
        super().__init__(config)

    def _constructor(
        self,
        pos_machine,
        sec,
        sr,
        stft_cfg={},
        metric="euclid",
        anc_neg_noise_prob=0.5,
        anc_neg_noise_mode="same",
        statex_prob=0.0,
        min_snr=2.5,
        max_snr=7.5,
        stretch_min_ratio_int=[0.5, 0.8],
        stretch_max_ratio_int=[1.2, 1.5],
        margin=1.0,
        margin_intra=-1.0,
    ) -> None:
        self.pos_machine = pos_machine
        self.stft_cfg = stft_cfg
        self.stft = STFT(**self.stft_cfg)
        spectrogram_size = self.stft(torch.randn(sec * sr)).shape
        self.embedding_size = 128
        self.extractor = STFT2dEncoderLayer(
            spectrogram_size, False, self.embedding_size
        )
        self.metric = metric
        assert self.metric in ["cosine", "euclid"]
        self.anc_neg_noise_prob = anc_neg_noise_prob
        self.anc_neg_noise_mode = anc_neg_noise_mode
        assert self.anc_neg_noise_mode in ["same", "partial"]
        self.statex_prob = statex_prob
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.stretch_min_ratio_int = stretch_min_ratio_int
        self.stretch_max_ratio_int = stretch_max_ratio_int
        self.margin = margin
        self.margin_intra = margin_intra

    def forward(self, wave: Tensor) -> Dict:
        """
        Args:
            x (Tensor): wave (B, T)
        """
        embedding = self.extractor(self.stft(wave))
        if self.metric == "cosine":
            return {"embedding": F.normalize(embedding, dim=1)}
        elif self.metric == "euclid":
            return {"embedding": embedding}

    def distance(self, e1, e2):
        """
        Args:
            e1, e2 (Tensor): embeddings (B, D)
        """
        if self.metric == "euclid":
            return torch.sqrt(torch.sum((e1 - e2) ** 2, dim=1))
        elif self.metric == "cosine":
            e1 = F.normalize(e1, dim=1)
            e2 = F.normalize(e2, dim=1)
            return 1 - torch.sum(e1 * e2, dim=1)

    def create_anc(self, target: Tensor, noise: Tensor):
        # Returns x or x+y1
        anchor = copy.deepcopy(target)
        noisy_idx = get_idx(self.anc_neg_noise_prob, len(target), target.device)
        anchor[noisy_idx] = mix_rand_snr(
            target[noisy_idx],
            noise[noisy_idx],
            self.min_snr,
            self.max_snr,
            self.stft.power,
        )
        return anchor, noisy_idx

    def create_pos(self, target: Tensor, noise: Tensor):
        # Returns x+y2
        perm = torch.randperm(len(target)).to(target.device)
        positive = mix_rand_snr(
            target, noise[perm], self.min_snr, self.max_snr, self.stft.power
        )
        return positive

    def create_neg(self, target: Tensor, noise: Tensor, noisy_idx):
        # Half of neg is processed sample
        target_modified = copy.deepcopy(target[: len(target) // 2])
        statex_idx = get_idx(self.statex_prob, len(target_modified), target.device)
        stretch_idx = ~statex_idx
        target_modified[statex_idx] = statex(target_modified[statex_idx], noise)
        target_modified[stretch_idx] = stretch(
            target_modified[stretch_idx],
            self.stretch_min_ratio_int,
            self.stretch_max_ratio_int,
        )
        # Half of neg is another sample
        perm = torch.randperm(len(target))[len(target) // 2 :].to(target.device)
        target_modified = torch.concat(
            [target_modified, copy.deepcopy(target[perm])], dim=0
        )
        # add noise (y1)
        if self.anc_neg_noise_mode == "partial":
            noisy_idx = get_idx(self.anc_neg_noise_prob, len(target), target.device)
        target_modified[noisy_idx] = mix_rand_snr(
            target_modified[noisy_idx],
            noise[noisy_idx],
            self.min_snr,
            self.max_snr,
            self.stft.power,
        )
        return target_modified

    def visualize_melspec(self, data, title, n=12):
        data = data.detach().cpu().numpy()
        fig, axes = plt.subplots(2, n // 2, figsize=(12, 8), tight_layout=True)
        axes = axes.flatten()
        for i in range(n):
            im = axes[i].imshow(
                amplitude_to_db(data[i]), origin="lower", cmap="viridis"
            )
            # axes[i].set_ylim
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, cax=cax)
        self.logger.experiment.add_figure(title, plt.gcf(), self.current_epoch)

    def training_step(self, batch, batch_idx):
        wave = batch.pop("wave")
        labels = batch  # just renamed
        is_pos = torch.Tensor(labels["machine"] == self.pos_machine)
        target = wave[is_pos == 1]
        noise = wave[is_pos == 0]

        with torch.no_grad():
            target_stft = self.stft(target)
            noise_stft = self.stft(noise)
            anc_stft, noisy_idx = self.create_anc(target_stft, noise_stft)
            pos_stft = self.create_pos(target_stft, noise_stft)
            neg_stft = self.create_neg(target_stft, noise_stft, noisy_idx)

        # if self.trainer.global_step % 32 == 0:
        #     self.visualize_melspec(target_stft, "Original Target")
        #     self.visualize_melspec(anc_stft, "Anchor")
        #     self.visualize_melspec(pos_stft, "Positive")
        #     self.visualize_melspec(neg_stft, "Negative")

        anc_emb = self.extractor(anc_stft)
        pos_emb = self.extractor(pos_stft)
        neg_emb = self.extractor(neg_stft)

        loss_dict = {}
        loss_dict["pull"] = self.distance(anc_emb, pos_emb)
        loss_dict["push"] = self.distance(anc_emb, neg_emb)
        loss_dict["main"] = torch.maximum(
            loss_dict["pull"] - loss_dict["push"] + self.margin,
            torch.zeros(len(target)).to(target.device),
        )
        if self.margin_intra > 0.0:
            loss_dict["main"] += torch.maximum(
                loss_dict["pull"] - self.margin_intra,
                torch.zeros(len(target)).to(target.device),
            )

        for key_ in loss_dict:
            loss_dict[key_] = torch.mean(loss_dict[key_])

        self.log_loss(torch.tensor(len(wave)).float(), f"train/batch_size", 1)
        for tag_, val_ in loss_dict.items():
            self.log_loss(val_, f"train/{tag_}", len(wave))

        return loss_dict["main"]
