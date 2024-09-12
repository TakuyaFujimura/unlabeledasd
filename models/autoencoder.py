# Copyright 2024 Takuya Fujimura

import sys
from typing import Dict

import torch
from torch import Tensor, nn

from .stft import STFT


class MLPAE(nn.Module):
    def __init__(self, n_frames: int, stft_cfg: dict, z_dim: int = 8, h_dim: int = 128):
        super().__init__()
        self.stft_cfg = stft_cfg
        self.stft = STFT(**self.stft_cfg)
        self.n_mels = (
            self.stft_cfg["n_mels"]
            if self.stft_cfg["use_mel"]
            else self.stft_cfg["n_fft"] // 2 + 1
        )
        self.n_frames = n_frames
        self.input_dim = self.n_frames * self.n_mels
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.embedding_size = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.z_dim),
            nn.BatchNorm1d(self.z_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.input_dim),
        )

    def extract_feat(self, wave):
        melspec = self.stft(wave)
        log_melspec = (
            20.0
            / self.stft.power
            * torch.log10(
                torch.maximum(
                    melspec, torch.tensor([sys.float_info.epsilon]).to(melspec.device)
                )
            )
        )
        B, F, T = log_melspec.shape
        remainder = T % self.n_frames
        if remainder > 0:
            log_melspec = torch.concat(
                [log_melspec[:, :, :-remainder], log_melspec[:, :, -self.n_frames :]],
                dim=-1,
            )
        consecutive = log_melspec.permute(0, 2, 1).reshape(-1, F * self.n_frames)
        return consecutive  # (B*ceil(T/n_frames), F*n_frames)

    def forward(self, x: Tensor) -> Dict:
        """
        Args:
            x (Tensor): (B, D)
        """
        assert len(x.shape) == 2 and x.shape[-1] == self.n_mels * self.n_frames
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return {"embedding": z, "reconstructed": x_hat}
