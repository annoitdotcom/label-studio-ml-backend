import math

import cv2
import numpy as np
import torch


def resize_image(img, args):
    height, width, _ = img.shape
    if height < width:
        new_width = (args["image_short_side"] / 32) * 32
        new_height = math.ceil(new_width / width * height / 32) * 32
    else:
        new_height = (args["image_short_side"] / 32) * 32
        new_width = math.ceil(new_height / height * width / 32) * 32

    new_size = (int(new_width), int(new_height))
    assert all(kk % 32 == 0 for kk in new_size)
    resized_img = cv2.resize(img, new_size)
    return resized_img


def preprocess_image(img, args):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
    original_shape = img.shape[:2]
    img = resize_image(img, args)
    img -= RGB_MEAN
    img /= 255.
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    return img, original_shape


def get_min_max_xy(all_xy):
    all_x = [x for idx, x in enumerate(all_xy) if idx % 2 == 0]
    all_y = [x for idx, x in enumerate(all_xy) if idx % 2 == 1]
    minx = min(all_x)
    miny = min(all_y)
    maxx = max(all_x)
    maxy = max(all_y)
    return (minx, miny, maxx, maxy)
