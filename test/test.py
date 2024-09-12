# Copyright 2024 Takuya Fujimura

import logging
import sys
from pathlib import Path

import hydra
from extract import extract_main
from lightning import seed_everything
from omegaconf import OmegaConf
from relpath import add_import_path
from score import score_main

add_import_path("..")
import pl_models
from utils import get_full_path


def get_ckpt_path(config):
    if config.all_mode:
        ckpt_path = get_full_path(
            f"{config.exp_root}/{config.name}/{config.version}/all/{config.checkpoint}",
        )
    else:
        ckpt_path = get_full_path(
            f"{config.exp_root}/{config.name}/{config.version}/{config.pos_machine}/{config.checkpoint}",
        )
    return ckpt_path


def load_plmodel(config, ckpt_path):
    ## Create model
    past_cfg = OmegaConf.load(Path(ckpt_path).parents[1] / "hparams.yaml")["config"]
    plmodel = eval(past_cfg.model.plmodel).load_from_checkpoint(ckpt_path)
    plmodel.to(config.device)
    logging.info("model was successfully loaded from ckpt_path")
    plmodel.eval()

    return plmodel, past_cfg


def make_dir(config, ckpt_path):
    # Make directory
    if config.all_mode:
        csv_dir = Path(ckpt_path).parents[2] / f"{config.pos_machine}/infer"
    else:
        csv_dir = Path(ckpt_path).parents[1] / f"infer"
    csv_dir.mkdir(exist_ok=True, parents=True)
    ver = config.feat_ver
    if ver is None:
        ver_list = [
            int(str(dirname).split("_")[-1]) for dirname in csv_dir.glob("version_*")
        ]
        ver = max(ver_list) + 1
    csv_dir = csv_dir / f"version_{ver}"
    if len([p for p in csv_dir.glob("*test_score.csv")]) > 0:
        print(f"already done: {csv_dir}")
        sys.exit(0)
    csv_dir.mkdir(exist_ok=True)
    OmegaConf.save(config, csv_dir / "hparams.yaml")
    return csv_dir


@hydra.main(version_base=None, config_path="../config_test", config_name="config")
def main(config):
    seed_everything(config.seed)
    ckpt_path = get_ckpt_path(config)
    csv_dir = make_dir(config, ckpt_path)
    basename = ckpt_path.split("/")[-1][: -len(".ckpt")]

    if config.extract:
        plmodel, past_cfg = load_plmodel(config, ckpt_path)
        extract_main(config, plmodel, basename, csv_dir, past_cfg)
    if config.score:
        score_main(config, basename, csv_dir)


if __name__ == "__main__":
    main()
