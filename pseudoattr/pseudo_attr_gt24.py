import argparse
import datetime
from pathlib import Path

import pandas as pd
from pseudoattr_utils import get_org_attr_machines
from pseudo_attr_gmm import org_attr
from relpath import add_import_path
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

add_import_path("..")
from utils import write_json


def check_df(df):
    for p_tgt, p_src in zip(df["target_file_name"], df["source_file_name"]):
        p_split_dict = {"tgt": p_tgt.split("_"), "src": p_src.split("_")}
        if p_split_dict["tgt"][5] != p_split_dict["src"][5]:
            return False
        for idx in ["tgt", "src"]:
            bool_list = [
                p_split_dict[idx][0] == "section",
                p_split_dict[idx][1] == "00",
                p_split_dict[idx][2] in ["source", "target"],
                p_split_dict[idx][3] == "train",
                p_split_dict[idx][4] == "normal",
            ]
            if not all(bool_list):
                return False
    return True


def pseudo_attr(machine: str, data_dir: str):
    df = pd.read_csv(f"ground_truth_attributes_train/{machine}.csv")
    path_list = [f"{data_dir}/{machine}/train/{p}" for p in df["target_file_name"]]
    attr_list = ["_".join(p.split("_")[6:]) for p in df["source_file_name"]]
    return path_list, attr_list


def convert_idx_dict(path_list, attr_list):
    idx_array = LabelEncoder().fit_transform(attr_list)
    idx_dict = {p: idx for p, idx in zip(path_list, idx_array.tolist())}
    return idx_dict


def save(
    save_dir,
    idx_dict,
    str_label_dict,
    pseudo_attr_machine,
    org_attr_machine,
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
        f.write(f"pseudo_attr_machine: {pseudo_attr_machine}\n")
        f.write(f"org_attr_machine: {org_attr_machine}\n")


def main(
    save_dir,
    data_dir,
    pseudo_attr_machine,
    org_attr_machine,
) -> None:

    if Path(f"label/{save_dir}").exists():
        print("Already done")
        return

    path_list = []
    attr_list = []

    # Get pseudo label
    for machine in tqdm(pseudo_attr_machine):
        path_list_, attr_list_ = pseudo_attr(machine, data_dir)
        path_list += path_list_
        attr_list += attr_list_

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
        pseudo_attr_machine,
        org_attr_machine,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--pseudo_attr_machines", type=str, nargs="+")
    args = parser.parse_args()
    org_attr_machines = get_org_attr_machines(args.data_dir, args.pseudo_attr_machines)

    print(args.pseudo_attr_machines)
    print(org_attr_machines)

    # failed_flg = False
    # for machine in args.pseudo_attr_machines:
    #     if not check_df(pd.read_csv(f"ground_truth_attributes_train/{machine}.csv")):
    #         print(f"Error: {machine}")
    #         failed_flg = True
    # if failed_flg:
    #     print("invalid csv file")
    #     exit()

    main(
        args.save_dir,
        args.data_dir,
        args.pseudo_attr_machines,
        org_attr_machines,
    )
