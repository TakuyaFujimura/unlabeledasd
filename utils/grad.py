# Copyright 2024 Takuya Fujimura

import torch


def grad_norm(module: torch.nn.Module):
    total_norm = 0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm
