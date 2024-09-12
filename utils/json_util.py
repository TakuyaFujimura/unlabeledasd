# Copyright 2024 Takuya Fujimura

import json


def write_json(json_path, data, indent=2):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=indent)


def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data
