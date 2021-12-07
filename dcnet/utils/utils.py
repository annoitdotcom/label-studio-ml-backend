import glob
import os
import subprocess
import zipfile

import cv2
import fuckit
import numpy as np
from PIL import Image

IMG_EXTENSIONS = ["*jpg", "*jpeg", "*png"]


def sync_big_files(root_dir):
    """Pull data from S3"""
    subprocess.run(
        "dvc pull",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        cwd=root_dir,
    )


def ensure_rgb(x):
    """x gotta be a PIL.Image"""
    return np.array(x.convert("RGB"))


@fuckit
def cast_image_to_array(x):
    """The fuckit module allow execution after failure just like normal multiple try catch clauses so we can stack return statements like this"""
    return ensure_rgb(Image.open(x))
    return ensure_rgb(Image.fromarray(x))
    return ensure_rgb(x)


def unzip_data(all_dataset):
    """
    Extract a zip file to its parent directory
    """
    print(all_dataset)
    for zip_path in all_dataset:
        folder = os.path.dirname(zip_path)
        with zipfile.ZipFile(zip_path) as zipper:
            for member in zipper.namelist():
                target_path = os.path.join(folder, member)
                if os.path.exists(target_path):
                    # If exist then not extracted
                    continue
                else:
                    # Else extract
                    zipper.extract(member, folder)


def resize_img(image, height=1024):

    width = int(image.shape[0] / (image.shape[1] / height))

    image = np.array(image, dtype=np.uint8)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    return image


def check_model_same_weights_torch(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True
