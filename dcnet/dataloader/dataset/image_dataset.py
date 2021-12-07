import math
import os

import cv2
import numpy as np

from dcnet.dataloader.dataset.dataset_base import DatasetBase


class ImageDataset(DatasetBase):
    """Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    """
    data_dir = None
    data_list = None
    processes = None

    def __init__(self, opt, processes, is_training):
        self.is_training = is_training
        if self.is_training:
            self.opt = opt.dataset.train
        else:
            self.opt = opt.dataset.validation
        # self.load_all(**kwargs)
        self.data_dir = self.opt.data_dir
        self.processes = processes

        # self.debug = cmd.get("debug", False)
        self.debug = False
        self.image_paths = []
        self.gt_paths = []
        self.get_all_samples()

    @classmethod
    def load_opt(cls, opt, processes, is_training):
        return cls(
            opt, processes, is_training
        )

    def get_all_samples(self):
        for i in range(len(self.data_dir)):

            if self.is_training:
                train_image_list = [image_name for image_name in os.listdir(
                    os.path.join(self.data_dir[i], "train_images"))]
                image_path = [self.data_dir[i]+"/train_images/" +
                              timg.strip() for timg in train_image_list]
                gt_path = [self.data_dir[i]+"/train_gts/" +
                           timg.strip()+".txt" for timg in train_image_list]
            else:
                test_image_list = [image_name for image_name in os.listdir(
                    os.path.join(self.data_dir[i], "test_images"))]
                image_path = [self.data_dir[i]+"/test_images/" +
                              timg.strip() for timg in test_image_list]
                gt_path = [self.data_dir[i]+"/test_gts/" +
                           timg.strip()+".txt" for timg in test_image_list]

            self.image_paths += image_path
            self.gt_paths += gt_path
        self.num_samples = len(self.image_paths)
        self.targets = self.load_ann()
        if self.is_training:
            assert len(self.image_paths) == len(self.targets)

    def load_ann(self):
        res = []
        for gt in self.gt_paths:
            lines = []
            reader = open(gt, "r").readlines()
            for line in reader:
                item = {}
                parts = line.strip().split(",")
                label = parts[-1]
                if label == "1":
                    label = "###"
                line = [i.strip("\ufeff").strip("\xef\xbb\xbf") for i in parts]
                if "icdar" in self.data_dir[0]:
                    poly = np.array(list(map(float, line[:8]))).reshape(
                        (-1, 2)).tolist()
                else:
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
        # print(data["filename"])

        debug_dir = "/mnt/ai_filestore/home/jason/sota_exp/DCNet/debug"

        # print("______________")
        # print("Ori shape", data["image"].shape)
        if self.processes is not None:
            for data_process in self.processes:
                # print(data_process.__class__.__name__, "Before",data.keys())
                data = data_process(data)
                # print(data_process.__class__.__name__, "After",data.keys())
                # "image", "gt", "mask", "thresh_map", "thresh_mask"-
                # print(data["image"].shape)

                # debug_dir = "/mnt/ai_filestore/home/jason/sota_exp/DCNet/debug"
                # if not self.is_training:
                #     try:
                #         print(data_process.__class__.__name__, data["image"].shape)
                #         # cv2.imwrite(os.path.join(debug_dir,  "{}_image.jpg".format(data_process.__class__.__name__) ), data["image"].reshape((640, 640, 3)))
                #     except Exception as ex:
                #         # cv2.imwrite(os.path.join(debug_dir,  "{}_image.jpg".format(data_process.__class__.__name__) ), data["image"].numpy())
                #         pass

        # print("_______________")

        # if not self.is_training:
        #     image = data["image"].reshape((640, 480, 3))
        #     gt = data["gt"].reshape((640, 480, 1))
        #     mask = data["mask"]
        #     # thresh_map = data["thresh_map"]
        #     # thresh_mask = data["thresh_mask"]
        #     debug_dir = "/mnt/ai_filestore/home/jason/sota_exp/DCNet/debug"
        #     print(image_path.split("/")[-1].split(".")[0] + "_image.jpg")
        #     print("Image shape", image.shape)
        #     print("Gt shape", gt.shape)
        #     print("---------------")

        #     cv2.imwrite(os.path.join(debug_dir,  image_path.split("/")[-1].split(".")[0] + "_image.jpg" ),image.cpu().numpy()*255)
        #     cv2.imwrite(os.path.join(debug_dir,  image_path.split("/")[-1].split(".")[0] + "_gt.jpg" ),gt * 255)
        #     cv2.imwrite(os.path.join(debug_dir,  image_path.split("/")[-1].split(".")[0] + "_mask.jpg" ),mask * 255)
            # cv2.imwrite(os.path.join(debug_dir,  image_path.split("/")[-1].split(".")[0] + "_thresh_map.jpg" ),thresh_map * 255)
            # cv2.imwrite(os.path.join(debug_dir,  image_path.split("/")[-1].split(".")[0] + "_thresh_mask.jpg" ),thresh_mask * 255)

        return data

    def __len__(self):
        return len(self.image_paths)
