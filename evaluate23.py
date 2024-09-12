# Copyright 2024 Takuya Fujimura

import argparse
import logging
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score

from utils import get_full_path


def myround(x):
    return Decimal(str(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


# Post_processes #####################################################


def get_post_processes(score_df):
    post_processes = [col for col in score_df.columns if col.split("-")[0] == "AS"]
    return post_processes


# Main ##############################################################


def make_result_df_col():
    result_df_col = []
    ## Added hmean result over domain and section
    hmean_col = [
        "hmean_official",
        "hmean_0_{d}_auc_only",
        "hmean_0_source_{a}_only",
        "hmean_0_target_{a}_only",
        "hmean_0_{d}_{a}_only",
        "hmean_0_{d}_auc_all",
        "hmean_0_{d}_{a}_all",
    ]
    # "hmean_{s}_{d}_{a}",

    result_df_col += hmean_col
    ## Added result per domain
    for all_only in ["only", "all"]:
        for domain in ["source", "target"]:
            result_df_col += [f"0_{domain}_auc_{all_only}"]
            result_df_col += [f"0_{domain}_pauc_{all_only}"]
    result_df_col += ["0_all_pauc_all"]

    return result_df_col, hmean_col


def fill_out_auc(eval_result_df, score_df, post_processes, hmean_key_list):
    # assert np.all(score_df["section_idx"].values == 0)
    for post_process in post_processes:
        for all_only in ["only", "all"]:
            for domain in ["source", "target"]:
                target_idx = score_df["is_target"].values == int(domain == "target")
                if all_only == "all":
                    target_idx = target_idx | (score_df["is_normal"].values == 0)
                for auc, max_fpr in zip(["auc", "pauc"], [None, 0.1]):
                    eval_name = f"0_{domain}_{auc}_{all_only}"
                    eval_score = roc_auc_score(
                        1 - score_df.loc[target_idx, "is_normal"],
                        score_df.loc[target_idx, post_process],
                        max_fpr=max_fpr,
                    )
                    eval_result_df.at[post_process, eval_name] = eval_score

                # 0_all_pauc_all
                eval_result_df.at[post_process, "0_all_pauc_all"] = roc_auc_score(
                    1 - score_df["is_normal"].values,
                    score_df[post_process].values,
                    max_fpr=0.1,
                )

        for hmean_name in hmean_key_list:
            assert hmean_name[:5] == "hmean"
            if hmean_name == "hmean_official":
                eval_result_df.at[post_process, hmean_name] = hmean(
                    np.array(
                        [
                            eval_result_df.at[post_process, "0_source_auc_all"],
                            eval_result_df.at[post_process, "0_target_auc_all"],
                            eval_result_df.at[post_process, "0_all_pauc_all"],
                        ]
                    )
                )
            else:
                hn_split = hmean_name.split("_")
                if hn_split[2] == "{d}":
                    if hn_split[3] == "{a}":
                        eval_result_df.at[post_process, hmean_name] = hmean(
                            np.array(
                                [
                                    eval_result_df.at[
                                        post_process, f"0_source_auc_{hn_split[-1]}"
                                    ],
                                    eval_result_df.at[
                                        post_process, f"0_target_auc_{hn_split[-1]}"
                                    ],
                                    eval_result_df.at[
                                        post_process, f"0_source_pauc_{hn_split[-1]}"
                                    ],
                                    eval_result_df.at[
                                        post_process, f"0_target_pauc_{hn_split[-1]}"
                                    ],
                                ]
                            )
                        )
                    elif hn_split[3] == "auc":
                        eval_result_df.at[post_process, hmean_name] = hmean(
                            np.array(
                                [
                                    eval_result_df.at[
                                        post_process, f"0_source_auc_{hn_split[-1]}"
                                    ],
                                    eval_result_df.at[
                                        post_process, f"0_target_auc_{hn_split[-1]}"
                                    ],
                                ]
                            )
                        )
                    else:
                        raise NotImplementedError()
                else:
                    if hn_split[3] == "{a}":
                        eval_result_df.at[post_process, hmean_name] = hmean(
                            np.array(
                                [
                                    eval_result_df.at[
                                        post_process,
                                        f"0_{hn_split[2]}_auc_{hn_split[-1]}",
                                    ],
                                    eval_result_df.at[
                                        post_process,
                                        f"0_{hn_split[2]}_pauc_{hn_split[-1]}",
                                    ],
                                ]
                            )
                        )
                    else:
                        raise NotImplementedError()

    eval_result_df = eval_result_df.reset_index().rename(
        columns={"index": "post_process"}
    )
    return eval_result_df


def confirmation(score_df):
    # assert np.all(score_df["section_idx"].values == 0)
    for domain in ["source", "target"]:
        n = ((score_df["is_target"] == int(domain == "target")).values).sum()
        if n != 100:
            logging.warning(f"{domain}-0: number of eval files is {n}")


def main(args):
    """Run scoring process for a machine"""
    # Read anomaly score DataFrame #######################################
    score_path = Path(get_full_path(args.score_path, args.all_mode))
    score_df = pd.read_csv(score_path)
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(
        filename=score_path.parent / "evaluate.log", level=logging.INFO, format=fmt
    )
    logging.info(f"Loaded from {score_path}.")
    post_processes = get_post_processes(score_df)

    # Make evaluation result DataFrame #####################################
    result_df_col, hmean_col = make_result_df_col()
    eval_result_df = pd.DataFrame(index=post_processes, columns=result_df_col)
    save_path = str(score_path).replace("_score.csv", "_result.csv")
    confirmation(score_df)

    # Calculation of AUC and pAUC per domain and section ###################
    eval_result_df = fill_out_auc(eval_result_df, score_df, post_processes, hmean_col)

    eval_result_df.to_csv(save_path, index=False)
    logging.info(f"Successfully saved at {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculates auc, pauc, hauc.")
    parser.add_argument("--score_path", type=str)
    parser.add_argument("--is_eval", action="store_true")
    parser.add_argument("--all_mode", action="store_true")
    parser.add_argument(
        "--eval_result_dir",
        type=str,
        help="exp_path/name/ver/<machine>/infer/version_0",
    )
    args = parser.parse_args()

    main(args)
