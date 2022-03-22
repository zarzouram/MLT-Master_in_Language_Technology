from typing import List, Dict, Union, Tuple
from numpy.typing import NDArray

from tqdm import tqdm
from pathlib import Path
from itertools import chain
import json

import h5py

import numpy as np

bbox = List[List[float]]
roi_dict = Dict[str, Union[str, bool, bbox]]
ignore_dict = Dict[str, bbox]


def load_jsonl(ann_path: Path) -> Dict[str, Union[ignore_dict, roi_dict]]:

    file_name = ann_path.name
    # load annotations files
    with open(ann_path, "r") as f:
        json_lines = f.readlines()

    data = {}
    for json_str in tqdm(json_lines, desc=f"loading: {file_name}",
                         unit="line"):
        json_obj = json.loads(json_str)

        img_name = json_obj["file_name"]
        data[img_name] = {
            "roi": list(chain.from_iterable(json_obj["annotations"])),
            "ignore": json_obj["ignore"]
        }

    return data


def write_h5_dataset(write_path: str,
                     X: NDArray,
                     y: NDArray,
                     names: Tuple[str, str],
                     units: Tuple[str, str],
                     attribute: List[List[float]] = None) -> None:

    with h5py.File(write_path, "w") as h5f:
        h5f.create_dataset(name=names[0],
                           data=X,
                           shape=np.shape(X),
                           dtype=units[0])

        h5f.create_dataset(name=names[1],
                           data=y,
                           shape=np.shape(y),
                           dtype=units[1])

        if attribute:
            h5f.attrs["MeanDev"] = attribute
