# Copyright 2024 Takuya Fujimura

import pickle


def write_pkl(pkl_path, data):
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)


def read_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data
