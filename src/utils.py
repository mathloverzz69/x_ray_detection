import json
import os
import shutil
import pathlib

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def load_data(data_dir_path: str, archive_name: str):
    archive_path = os.path.join(data_dir_path, archive_name)
    shutil.unpack_archive(archive_path, data_dir_path)

    annot_cat = {}
    img_cat = {}
    for split in ("train", "valid", "test"):
        data_dir = pathlib.Path(os.path.join(
            data_dir_path, split)).with_suffix('')

        # save annotations
        ann_adress = os.path.join(data_dir, "_annotations.coco.json")
        with open(ann_adress) as f:
            annotation = json.load(f)

        annot_cat[f"{split}"] = annotation

        image_files = list(data_dir.glob('*.jpg'))
        img_cat[f"{split}"] = image_files
        print(f"{split} data:", len(image_files))  # training images

    return annot_cat, img_cat
