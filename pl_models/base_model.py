# Copyright 2024 Takuya Fujimura

import lightning.pytorch as pl
import numpy as np
import torch
from hydra.utils import instantiate

from utils import grad_norm


class BasePLModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.optim_cfg = config.model.optim_cfg
        self.scheduler_cfg = config.model.scheduler_cfg

        self.grad_clipper = None
        self.grad_every_n_steps = 25

        self.valid_melspec_gt_list = []
        self._constructor(**config.model.model_cfg)

    def _constructor(self):
        pass

    def log_loss(self, loss, log_name, batch_size):
        if isinstance(loss, torch.Tensor) and loss.numel() > 1:
            return
        if isinstance(loss, np.ndarray) and loss.size > 1:
            return
        self.log(log_name, loss, prog_bar=True, batch_size=batch_size, sync_dist=True)

    def on_after_backward(self):
        clipping_threshold = None
        if self.grad_clipper is not None:
            grad_norm_val, clipping_threshold = self.grad_clipper(self)
        else:
            grad_norm_val = grad_norm(self)
        if self.trainer.global_step % self.grad_every_n_steps == 0:
            if clipping_threshold is None:
                clipped_norm_val = grad_norm_val
            else:
                clipped_norm_val = min(grad_norm_val, clipping_threshold)
            opt = self.trainer.optimizers[0]
            current_lr = opt.state_dict()["param_groups"][0]["lr"]
            self.logger.log_metrics(
                {
                    "grad/norm": grad_norm_val,
                    "grad/clipped_norm": clipped_norm_val,
                    "grad/lr": current_lr,
                    "grad/step_size": current_lr * clipped_norm_val,
                },
                step=self.trainer.global_step,
            )

    def configure_optimizers(self):
        optimizer = instantiate({"params": self.parameters(), **self.optim_cfg})
        if self.scheduler_cfg is not None:
            scheduler = instantiate({"optimizer": optimizer, **self.scheduler_cfg})
            lr_scheduler = {"scheduler": scheduler, "interval": "step"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()
