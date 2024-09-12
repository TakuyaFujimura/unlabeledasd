# Copyright 2024 Takuya Fujimura

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SCAdaCos(nn.Module):
    def __init__(
        self,
        embed_size,
        n_classes=None,
        n_subclusters=1,
        eps=1e-7,
        trainable=False,
        reduction="mean",
        dynamic=True,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_subclusters = n_subclusters
        self.s_init = np.sqrt(2) * np.log(n_classes * n_subclusters - 1)
        self.eps = eps

        # Weight initialization
        self.W = nn.Parameter(
            torch.Tensor(embed_size, n_classes * n_subclusters), requires_grad=trainable
        )
        nn.init.xavier_uniform_(self.W.data)

        # Scale factor
        self.s = nn.Parameter(torch.tensor(self.s_init), requires_grad=False)
        self.reduction = reduction
        self.loss_fn = nn.CrossEntropyLoss(reduction=self.reduction)

        self.dynamic = dynamic

    def forward(self, x, t):
        t_orig = t.clone()
        t = t.repeat(1, self.n_subclusters)

        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.W, p=2, dim=0)

        # Dot product
        logits = torch.mm(x, W)
        theta = torch.acos(torch.clamp(logits, -1.0 + self.eps, 1.0 - self.eps))

        if self.training and self.dynamic:
            with torch.no_grad():
                max_s_logits = torch.max(self.s * logits)
                B_avg = torch.exp(self.s * logits - max_s_logits)  # re-scaling trick
                B_avg = torch.mean(torch.sum(B_avg, dim=1))
                theta_class = torch.sum(t * theta, dim=1)
                theta_med = torch.median(theta_class)
                self.s.data = max_s_logits + torch.log(B_avg)  # re-scaling trick
                self.s.data /= (
                    torch.cos(min(torch.tensor(np.pi / 4), theta_med)) + self.eps
                )

        logits *= self.s
        prob = F.softmax(logits, dim=1)
        prob = prob.view(
            -1, self.n_classes, self.n_subclusters
        )  # (B, C, n_subclusters)
        prob = torch.sum(prob, dim=2)  # (B, C)
        loss = self.loss_fn(torch.log(prob), t_orig)
        return loss

