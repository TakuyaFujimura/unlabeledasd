import argparse
import glob
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torchaudio
from pseudoattr_utils import extract
from omegaconf import OmegaConf
from relpath import add_import_path
from torch.utils.data import DataLoader, Dataset

add_import_path("..")

import pl_models
from utils import get_path_glob


class WavPathDataset(Dataset):
    def __init__(self, path_list, past_cfg):
        super().__init__()
        self.path_list = path_list
        assert past_cfg.sampling_rate == 16000
        self.crop_len = past_cfg.datamodule.sec * past_cfg.sampling_rate

    def __getitem__(self, idx):
        path = self.path_list[idx]
        wave, fs = torchaudio.load(path)
        assert fs == 16000 and wave.shape[0] == 1
        wave = wave[0]
        wave = wave.tile(int(np.ceil(self.crop_len / len(wave))))
        wave = wave[: self.crop_len]
        return wave, path

    def __len__(self):
        """Return dataset length."""
        return len(self.path_list)


def load_plmodel(ckpt_path, device):
    # Create model
    past_cfg = OmegaConf.load(Path(ckpt_path).parents[1] / "hparams.yaml")["config"]
    plmodel = eval(past_cfg.model.plmodel).load_from_checkpoint(ckpt_path)
    plmodel.to(device)
    plmodel.eval()
    return plmodel, past_cfg


def make_loader(data_dir, past_cfg, machine):
    path_list = glob.glob(f"{data_dir}/{machine}/*/*.wav")
    print(f"len(dataloader) was {len(path_list)}")
    loader = DataLoader(
        dataset=WavPathDataset(path_list, past_cfg),
        batch_size=32,
        num_workers=0,
        shuffle=False,
    )
    return loader


def main() -> None:
    pl.seed_everything(0, workers=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--ckpt_cond", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--machines", type=str, nargs="+")
    parser.add_argument("--all_ckpt", action="store_true")
    args = parser.parse_args()

    dcase = args.model_path.split("/")[-2]
    if dcase not in ["dcase2023", "dcase2024"]:
        raise NotImplementedError("Only dcase2023 and dcase2024 are supported")
    if args.data_dir.split("/")[-3] != dcase:
        raise ValueError("data_dir and model_path should be in the same dcase setting")

    model_name = args.model_path.split("/")[-1]
    save_dir = Path(f"embed/{dcase}/{model_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    for m in args.machines:
        if args.all_ckpt:
            ckpt_path = get_path_glob(
                f"{args.model_path}/all/checkpoints/{args.ckpt_cond}"
            )
        else:
            ckpt_path = get_path_glob(
                f"{args.model_path}/{m}/checkpoints/{args.ckpt_cond}"
            )
        ckpt_path = Path(ckpt_path)
        plmodel, past_cfg = load_plmodel(ckpt_path, args.device)
        loader = make_loader(args.data_dir, past_cfg, m)
        df = extract(plmodel, loader, args.device)
        df.to_csv(save_dir / f"{ckpt_path.stem}_{m}.csv")


if __name__ == "__main__":
    main()
