import codecs
import json
import math
import os

import cv2
import numpy as np
from PIL import Image

from dcnet.dataloader.dataset.dataset_base import DatasetBase


class DatapileDataset(DatasetBase):
    """Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    """
    data_dir = None
    data_list = None
    processes = None

    def __init__(self, opt, processes, is_training, data_paths=None):
        self.is_training = is_training
        if self.is_training:
            self.opt = opt.dataset.train
        else:
            self.opt = opt.dataset.validation
        # self.load_all(**kwargs)

        # self.data_dir = data
        self.processes = processes

        # self.debug = cmd.get("debug", False)
        self.debug = False
        self.image_paths = []
        self.gt_paths = []

        if data_paths:
            (self.image_paths, self.gt_paths) = data_paths
        else:
            self.data_dir = self.opt.data_dir
            self.get_all_samples()

        self.num_samples = len(self.image_paths)
        self.targets = self.load_ann()

        if self.is_training:
            assert len(self.image_paths) == len(self.targets)

    @classmethod
    def load_opt(cls, opt, processes, is_training, data_paths=None):
        return cls(
            opt, processes, is_training, data_paths
        )

    def get_all_samples(self):
        for i in range(len(self.data_dir)):
            image_dir = os.path.join(self.data_dir[i], "images")
            label_dir = os.path.join(self.data_dir[i], "labels")

            image_path = []
            label_path = []

            for file in os.listdir(image_dir):
                image_name = file.split(".")[0]

                image_path.append(os.path.join(image_dir, file))
                label_path.append(os.path.join(
                    label_dir, image_name + ".json"))

            self.image_paths += image_path
            self.gt_paths += label_path

    def parse_data(self, image_path, label_path, debug_dir=""):
        lines = []

        img = np.asarray(Image.open(image_path))
        with codecs.open(label_path, "r", encoding="utf-8-sig") as fi:
            label_data = json.load(fi)

        # regions = label_data["attributes"]["_via_img_metadata"]["regions"]
        fn = os.path.basename(image_path).split(".")[0]
        key_name = list(filter(lambda k: fn in k, list(label_data.keys())))[-1]
        regions = label_data[key_name]["regions"]

        for idx, region in enumerate(regions):
            line = ""
            type_fm_key = "other"
            if region["region_attributes"].get("key_type", ""):
                type_fm_key = region["region_attributes"].get("key_type", "")

            if region["shape_attributes"]["name"] == "polygon":
                all_x, all_y = self.__getallxy(region)

                for i in range(len(all_x)):
                    line += "{},{},".format(all_x[i], all_y[i])
                    x1, y1 = all_x[i], all_y[i]
                    x2, y2 = all_x[(i + 1) % len(all_x)
                                   ], all_y[(i + 1) % len(all_y)]
                    if debug_dir:
                        img = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                if debug_dir:
                    org = (all_x[0], all_y[0])
                    img = cv2.putText(
                        img, type_fm_key, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                line += type_fm_key
                lines.append(line)

            elif region["shape_attributes"]["name"] == "rect":
                x, y, width, height = self.__getxywh(region)
                minx, miny, maxx, maxy = x, y, x + width, y + height
                line = "{},{},{},{},{},{},{},{},{}\n".format(
                    minx, miny, maxx, miny, maxx, maxy, minx, maxy, type_fm_key)

                if debug_dir:
                    org = (minx, miny)
                    img = cv2.putText(
                        img, type_fm_key, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    img = cv2.rectangle(img, (minx, miny),
                                        (maxx, maxy), (0, 0, 255), 3)
                lines.append(line)

        if debug_dir:
            image_name = image_path.replace("\\", "/").split("/")[-1]
            Image.fromarray(img).save(
                os.path.join(debug_dir, image_name + ".png"))

        return lines

    def load_ann(self):
        res = []
        for i in range(len(self.gt_paths)):
            lines = []
            data = self.parse_data(self.image_paths[i], self.gt_paths[i])
            for line in data:

                item = {}
                parts = line.strip().split(",")
                label = parts[-1]
                if label == "1":
                    label = "###"
                line = [i.strip("\ufeff").strip("\xef\xbb\xbf") for i in parts]

                num_points = math.floor((len(line) - 1) / 2) * 2
                poly = np.array(list(map(float, line[:num_points]))).reshape(
                    (-1, 2)).tolist()

                item["poly"] = poly
                item["text"] = label
                lines.append(item)
            res.append(lines)
        return res

    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        image_path = self.image_paths[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype("float32")
        if self.is_training:
            data["filename"] = image_path
            data["data_id"] = image_path
        else:
            data["filename"] = image_path.split("/")[-1]
            data["data_id"] = image_path.split("/")[-1]

        data["image"] = img
        target = self.targets[index]
        data["lines"] = target

        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)
        return data

    def __len__(self):
        return len(self.image_paths)

    def __getxywh(self, shape_attr):
        x = int(shape_attr["shape_attributes"]["x"])
        y = int(shape_attr["shape_attributes"]["y"])
        width = int(shape_attr["shape_attributes"]["width"])
        height = int(shape_attr["shape_attributes"]["height"])

        return x, y, width, height

    def __getallxy(self, shape_attr):
        all_x = shape_attr["shape_attributes"]["all_points_x"]
        all_y = shape_attr["shape_attributes"]["all_points_y"]

        return all_x, all_y
