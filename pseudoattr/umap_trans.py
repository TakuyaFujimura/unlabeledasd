import argparse
import glob
from pathlib import Path

import pandas as pd
import umap


def trans_to_u(e_df, metric):
    ecols = [col for col in e_df.keys() if col[0] == "e" and int(col[1:]) >= 0]
    print(len(ecols))
    umap_model = umap.UMAP(random_state=0, metric=metric)
    u = umap_model.fit_transform(e_df[ecols].values)
    u_df = pd.DataFrame.from_dict(
        {"path": e_df.path.values, "u0": u[:, 0], "u1": u[:, 1]}
    )
    return u_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--ckpt_cond", type=str)
    parser.add_argument("--metric", type=str, default="euclidean")
    args = parser.parse_args()
    model_path = "/".join(args.model_path.split("/")[-2:])
    save_dir = Path(f"umap_embed/{model_path}")
    save_dir.mkdir(parents=True, exist_ok=True)
    embed_path_list = glob.glob(f"embed/{model_path}/*{args.ckpt_cond}*.csv")
    print(len(embed_path_list))
    for embed_path in embed_path_list:
        embed_path = Path(embed_path)
        e_df = pd.read_csv(embed_path)
        u_df = trans_to_u(e_df, args.metric)
        u_df.to_csv(save_dir / f"{embed_path.stem}_{args.metric}.csv")


if __name__ == "__main__":
    main()
