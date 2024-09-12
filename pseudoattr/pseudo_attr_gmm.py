import argparse
import datetime
import glob
from pathlib import Path

from clustering_gmm import gmm_clustering
from pseudoattr_utils import extract_emb, get_org_attr_machines, get_train_dom_csv
from relpath import add_import_path
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

add_import_path("..")
from utils import get_path_glob, write_json


def pseudo_attr(
    machine: str, ckpt_path: str, n_cluster_dict: dict, compression: bool, ic: str
):
    """Obtain pseudo attribute by bottomup ensemble clustering

    Args:
        machine (str): machine
        ckpt_path (str): ckpt_path
        n_cluster_dict (dict): number of cluster for each domain

    Returns:
        path_list: list of path
        multi_attr_list: list of "machine-domain-attr"
    """
    all_path_list = []
    all_attr_list = []
    info_df_dict = {}
    for dom, n_cluster in n_cluster_dict.items():
        if compression:
            df = get_train_dom_csv(
                get_path_glob(f"umap_{ckpt_path}_{machine}*.csv"), dom
            )
            Z = extract_emb(df, keyword="u")
        else:
            df = get_train_dom_csv(get_path_glob(f"{ckpt_path}_{machine}.csv"), dom)
            Z = extract_emb(df, keyword="e")

        path_list = df.path.values.tolist()

        idx_list, info_df = gmm_clustering(Z, n_cluster, ic)
        info_df_dict[dom] = info_df
        all_path_list += path_list
        all_attr_list += [f"{machine}-{dom}-{idx}" for idx in idx_list]
    return all_path_list, all_attr_list, info_df_dict["source"], info_df_dict["target"]


def org_attr(machine: str, data_dir: str):
    """obtain original attribute from path

    Args:
        machine (str): machine
        data_dir (str): data_dir

    Returns:
        path_list: path list
        multi_attr: list of "machine-domain-orgattr"
    """
    path_list = glob.glob(f"{data_dir}/{machine}/train/*.wav")
    attr_list = []
    for p in path_list:
        split_path_list = p.split("/")[-1].split("_")
        dom = split_path_list[2]
        attr = "_".join(split_path_list[6:])
        attr_list += [f"{machine}-{dom}-{attr}"]
    return path_list, attr_list


def convert_idx_dict(path_list, attr_list):
    idx_array = LabelEncoder().fit_transform(attr_list)
    idx_dict = {p: idx for p, idx in zip(path_list, idx_array.tolist())}
    return idx_dict


def save(
    save_dir,
    idx_dict,
    str_label_dict,
    ckpt_path,
    n_cluster_dict,
    pseudo_attr_machine,
    org_attr_machine,
    compression,
    ic,
    info_df_dict,
):
    Path(f"label/{save_dir}").mkdir(exist_ok=True, parents=True)
    # JSON
    write_json(f"label/{save_dir}/idx.json", idx_dict)
    write_json(f"label/{save_dir}/strlabel.json", str_label_dict)
    # Text of information
    with open(
        f"label/{save_dir}/info.txt",
        "w",
    ) as f:
        f.write(f"{datetime.datetime.today()}\n")
        f.write(f"ckpt_path: {ckpt_path}\n")
        f.write(f"number of cluster: {n_cluster_dict}\n")
        f.write(f"pseudo_attr_machine: {pseudo_attr_machine}\n")
        f.write(f"org_attr_machine: {org_attr_machine}\n")
        f.write(f"compression: {compression}\n")
        f.write(f"ic: {ic}\n")

    for m in pseudo_attr_machine:
        for dom in ["source", "target"]:
            info_df_dict[f"{m}_{dom}"].to_csv(
                f"label/{save_dir}/{m}_{dom}.csv", index=False
            )


def main(
    save_dir,
    data_dir,
    pseudo_attr_machine,
    org_attr_machine,
    ckpt_path,
    n_cluster_source,
    n_cluster_target,
    compression,
    ic,
) -> None:

    if Path(f"label/{save_dir}").exists():
        print("Already done")
        return

    n_cluster_dict = {"source": n_cluster_source, "target": n_cluster_target}
    path_list = []
    attr_list = []
    info_df_dict = {}

    # Get pseudo label
    for machine in tqdm(pseudo_attr_machine):
        path_list_, attr_list_, info_df_so, info_df_ta = pseudo_attr(
            machine, ckpt_path, n_cluster_dict, compression, ic
        )
        path_list += path_list_
        attr_list += attr_list_
        info_df_dict[f"{machine}_source"] = info_df_so
        info_df_dict[f"{machine}_target"] = info_df_ta

    # Get original label
    for machine in tqdm(org_attr_machine):
        path_list_, attr_list_ = org_attr(machine, data_dir)
        path_list += path_list_
        attr_list += attr_list_

    idx_dict = convert_idx_dict(path_list, attr_list)
    str_label_dict = {p: l for p, l in zip(path_list, attr_list)}

    # save
    save(
        save_dir,
        idx_dict,
        str_label_dict,
        ckpt_path,
        n_cluster_dict,
        pseudo_attr_machine,
        org_attr_machine,
        compression,
        ic,
        info_df_dict,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--pseudo_attr_machines", type=str, nargs="+")
    parser.add_argument("--n_cluster_source", type=int, default=16)
    parser.add_argument("--n_cluster_target", type=int, default=2)
    parser.add_argument("--compression", action="store_true")
    parser.add_argument("--ic", type=str)
    args = parser.parse_args()
    org_attr_machines = get_org_attr_machines(args.data_dir, args.pseudo_attr_machines)

    print(args.pseudo_attr_machines)
    print(org_attr_machines)

    main(
        args.save_dir,
        args.data_dir,
        args.pseudo_attr_machines,
        org_attr_machines,
        args.ckpt_path,
        args.n_cluster_source,
        args.n_cluster_target,
        args.compression,
        args.ic,
    )
