# Copyright 2024 Takuya Fujimura

import logging
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar

import pl_models
from datasets import ASDDataModule, ASDGenDataModule


class NaNCheckCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._check_for_nan(outputs):
            logging.warning("NaN detected in training batch, stopping training.")
            trainer.should_stop = True

    @staticmethod
    def _check_for_nan(outputs):
        if isinstance(outputs, torch.Tensor):
            return torch.isnan(outputs).any().item()
        elif isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and torch.isnan(value).any().item():
                    return True
        return False


def make_tb_logger(cfg):
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=cfg.path.exp_root,
        name=cfg.name,
        version=cfg.version,
        sub_dir=cfg.pos_machine,
    )
    return tb_logger


def make_trainer(cfg, tb_logger):
    callback_list = [NaNCheckCallback()]
    for key_, cfg_ in cfg.callback_opts.items():
        callback_list.append(
            ModelCheckpoint(**{**cfg_, "dirpath": tb_logger.log_dir + "/checkpoints"})
        )
    callback_list.append(TQDMProgressBar(refresh_rate=cfg.refresh_rate))
    trainer = instantiate(
        {
            **cfg.trainer,
            "callbacks": callback_list,
            "logger": tb_logger,
            "check_val_every_n_epoch": cfg.every_n_epochs_valid,
        }
    )
    return trainer


def check_cfg(cfg):
    assert cfg.name == cfg.datamodule.dcase


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg) -> None:
    check_cfg(cfg)
    if not cfg.trainer.deterministic:
        logging.warning("Not deterministic!!!")
    exp_name = HydraConfig().get().run.dir
    logging.info(f"Start experiment: {exp_name}")
    logging.info(f"version: {cfg.version}")
    seed_everything(cfg.seed, workers=True)
    torch.autograd.set_detect_anomaly(False)

    tb_logger = make_tb_logger(cfg)
    if Path(tb_logger.log_dir + "/checkpoints").exists():
        logging.warning("already done")
        return

    # before creating model, we should set the steps_per_epoch of mixup
    # using datamodule paramters
    if getattr(cfg.datamodule, "gen", False):
        dm = ASDGenDataModule(cfg)
    else:
        dm = ASDDataModule(cfg)
    dm.setup(None)

    logging.info("Create new model")
    plmodel = eval(cfg.model.plmodel)(cfg)  # save hyperparameters in pl_model
    logging.info(f"Number of parameters: {count_parameters(plmodel)}")
    trainer = make_trainer(cfg, tb_logger)

    logging.info("Start Training")
    trainer.fit(plmodel, dm.train_dataloader(), dm.val_dataloader())


if __name__ == "__main__":
    main()
