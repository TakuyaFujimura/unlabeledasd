# Copyright 2024 Takuya Fujimura
# ground_truth_attributes_test_23 is obtained from https://github.com/nttcslab/dcase2023_task2_evaluator/tree/main/ground_truth_attributes
# ground_truth_attributes_test_24 is obtained from https://github.com/nttcslab/dcase2024_task2_evaluator/tree/main/ground_truth_attributes

import argparse
import csv
from pathlib import Path

from relpath import rel2abs
from tqdm import tqdm


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(description="Add ground truth label")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
    )
    args = parser.parse_args()
    for dcase in ["23", "24"]:
        for csv_path in Path(rel2abs(f"./ground_truth_attributes_test_{dcase}")).glob("*.csv"):
            add_gt_info_test(csv_path, Path(args.data_dir) / f"dcase20{dcase}/all/raw")


def add_gt_info_test(csv_path: Path, data_dir: Path):
    """
    Args:
        csv_path (Path): _description_
        data_dir (Path): _description_
    """
    machine = csv_path.stem.split("_")[2]
    with open(csv_path, mode="r") as file:
        csv_reader = csv.reader(file)
        for row in tqdm(csv_reader):
            old_path = data_dir / f"{machine}/test/{row[0]}"
            if ".wav" in row[-1]:
                new_path = data_dir / f"{machine}/test/{row[-1]}"
            else:
                new_path = data_dir / f"{machine}/test/{row[-1]}.wav"
            old_path.rename(new_path)


if __name__ == "__main__":
    main()
