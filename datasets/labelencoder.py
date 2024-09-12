# Copyright 2024 Takuya Fujimura
from pathlib import Path

import torch
from sklearn.preprocessing import LabelEncoder

from .torch_dataset import get_basic_label


class Labeler:
    def __init__(self, label_name, pretrained_dict=None, mode="single"):
        assert label_name[:7] == "onehot_"
        self.label_name = label_name[7:]
        self.pseudo_label_dict = {}
        self.pretrained_dict = pretrained_dict
        self.mode = mode

    def get_label(self, path):
        if self.label_name in ["machine"]:
            return get_basic_label(path, self.label_name)
        elif self.label_name in ["machine_attr"]:
            return "-".join(
                [
                    str(get_basic_label(path, "machine")),
                    str(get_basic_label(path, "attr")),
                ]
            )
        elif self.label_name in ["machine_attr_is_target"]:
            return "-".join(
                [
                    str(get_basic_label(path, "machine")),
                    str(get_basic_label(path, "attr")),
                    str(get_basic_label(path, "is_target")),
                ]
            )
        elif self.label_name in ["machine_is_target"]:
            return "-".join(
                [
                    str(get_basic_label(path, "machine")),
                    str(get_basic_label(path, "is_target")),
                ]
            )
        elif self.pretrained_dict is not None:
            return self.pretrained_dict[path]
        else:
            raise NotImplementedError()

    def fit(self, path_list):
        if self.mode == "single":
            all_label = [self.get_label(p) for p in path_list]
        elif self.mode == "multi":
            # flatten and concatenate all labels
            all_label = []
            for p in path_list:
                all_label += self.get_label(p)  # list of K labels
        else:
            raise NotImplementedError()

        self.le = LabelEncoder()
        self.le.fit(all_label)
        self.num = len(self.le.classes_)

    def trans(self, path_list):
        if self.mode == "single":
            label_idx = self.le.transform([self.get_label(p) for p in path_list])
            label_onehot = torch.nn.functional.one_hot(
                torch.from_numpy(label_idx),
                num_classes=self.num,
            ).float()
            return {"idx": label_idx, "onehot": label_onehot}
        elif self.mode == "multi":
            multiple_label_idx = []
            for p in path_list:
                multiple_label_idx += [self.le.transform(self.get_label(p)).tolist()]
                # Each path has K labels
            multiple_label_idx = torch.tensor(multiple_label_idx)  # B, K

            multiple_onehot_list = [
                torch.nn.functional.one_hot(multiple_idx, num_classes=self.num).float()
                for multiple_idx in multiple_label_idx
            ]
            # (B, K, self.num)

            return {
                "idx": multiple_label_idx,
                "onehot": torch.stack(multiple_onehot_list, dim=0),
            }
        else:
            raise NotImplementedError()
