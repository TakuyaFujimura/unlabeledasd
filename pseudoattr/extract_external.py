import argparse
import glob
import random
from pathlib import Path

import numpy as np
import resampy
import torch
import torchaudio
from pseudoattr_utils import extract
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WavPathDataset18(Dataset):
    def __init__(self, path_list):
        super().__init__()
        self.path_list = path_list
        self.crop_len = 18 * 16000

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


def make_loader(data_dir, machine):
    path_list = glob.glob(f"{data_dir}/{machine}/*/*.wav")
    print(f"len(dataloader) was {len(path_list)}")
    loader = DataLoader(
        dataset=WavPathDataset18(path_list),
        batch_size=32,
        num_workers=0,
        shuffle=False,
    )
    return loader


class openl3Model(torch.nn.Module):
    def __init__(self, hp):
        super().__init__()
        # input_repr="mel128", content_type="env", embed_size=512
        # pip install torchopenl3
        # https://github.com/turian/torchopenl3.git

        self.hp = hp
        if self.hp == "env512mel128":
            input_repr = "mel128"
            content_type = "env"
            embed_size = 512
        else:
            raise NotImplementedError()
            # assert self.embed_size in [512, 6144]
            # assert content_type in ["env", "music"]
            # assert input_repr in ["linear", "mel128", "mel256"]

        from torchopenl3 import get_audio_embedding
        from torchopenl3.models import load_audio_embedding_model

        self.embedding_size = int(embed_size)

        self.model = load_audio_embedding_model(
            input_repr=input_repr,
            content_type=content_type,
            embedding_size=self.embedding_size,
        )
        self.get_audio_embedding = get_audio_embedding

    def forward(self, wave_tensor_cpu):
        TARGET_SR = 48000
        sr_list = [TARGET_SR for _ in range(len(wave_tensor_cpu))]
        wave_list = [
            resampy.resample(
                wave.numpy(),
                sr_orig=16000,
                sr_new=TARGET_SR,
                filter="kaiser_best",
            )
            for wave in wave_tensor_cpu
        ]
        emb_list, ts_list = self.get_audio_embedding(
            wave_list, sr_list, model=self.model
        )
        assert emb_list[0].shape[-1] == self.embedding_size
        return {"embedding": torch.stack(emb_list).mean(dim=1)}  # (B, T, D) -> (B, D)


class pannsModel(torch.nn.Module):
    def __init__(self, hp):
        super().__init__()
        # pip install panss_inference
        # https://github.com/qiuqiangkong/panns_inference

        from panns_inference import AudioTagging

        # audio.shape is (batch_size, segment_samples)
        self.device = "cuda:0"
        self.model = AudioTagging(
            checkpoint_path="./Cnn14_mAP=0.431.pth", device="cuda"
        )

        self.hp = hp
        assert self.hp == "CNN14"
        self.embedding_size = 2048  # int(embed_size)

    def forward(self, wave_tensor_cpu):
        TARGET_SR = 32000
        wave_list = [
            resampy.resample(
                wave.numpy(),
                sr_orig=16000,
                sr_new=TARGET_SR,
                filter="kaiser_best",
            )
            for wave in wave_tensor_cpu
        ]
        clipwise_output, embedding = self.model.inference(
            torch.tensor(np.array(wave_list)).to(self.device)
        )
        assert embedding.shape[-1] == self.embedding_size
        return {"embedding": embedding}


def load_model(model_name, model_hp):
    if model_name == "openl3":
        model = openl3Model(model_hp)
    elif model_name == "panns":
        model = pannsModel(model_hp)
    else:
        raise NotImplementedError()
    return model


def main() -> None:
    seed_everything(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_hp", type=str)
    parser.add_argument("--machines", type=str, nargs="+")
    args = parser.parse_args()

    dcase = args.data_dir.split("/")[-3]
    save_dir = Path(f"embed/{dcase}/{args.model_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    for m in args.machines:
        loader = make_loader(args.data_dir, m)
        model = load_model(args.model_name, args.model_hp)
        df = extract(model, loader, "cpu")
        df.to_csv(save_dir / f"{args.model_hp}_{m}.csv")


if __name__ == "__main__":
    main()
