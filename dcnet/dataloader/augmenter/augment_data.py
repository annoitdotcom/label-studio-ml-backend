import math

import cv2
import imgaug
import numpy as np

from dcnet.dataloader.augmenter.augment_base import AugmenterBase
from dcnet.dataloader.augmenter.augmenter import AugmenterBuilder


class AugmentData(AugmenterBase):

    def __init__(self, opt, is_training):
        if is_training:
            self.opt = opt.dataset.train
        else:
            self.opt = opt.dataset.validation
        self.is_training = is_training
        self.augmenter_args = self.opt.augmenter_args
        self.keep_ratio = self.opt.keep_ratio
        self.only_resize = self.opt.only_resize

        print("AugmentData:")
        print("keep_ratio", self.keep_ratio)
        print("only_resize", self.only_resize)
        print("augmenter_args", self.augmenter_args)

        self.augmenter = AugmenterBuilder().build(self.augmenter_args)

    def may_augment_annotation(self, aug, data):
        pass

    @classmethod
    def load_opt(cls, opt, is_training=True):
        return cls(opt, is_training)

    def resize_image(self, image):
        origin_height, origin_width, _ = image.shape
        resize_shape = self.augmenter_args[0][1]
        height = resize_shape["height"]
        width = resize_shape["width"]
        if self.keep_ratio:
            width = origin_width * height / origin_height
            N = math.ceil(width / 32)
            width = N * 32
        image = cv2.resize(image, (width, height))
        return image

    def __call__(self, data):
        image = data["image"]
        aug = None
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            if self.only_resize:
                data["image"] = self.resize_image(image)
            else:
                data["image"] = aug.augment_image(image)
            self.may_augment_annotation(aug, data, shape)

        filename = data.get("filename", data.get("data_id", ""))
        data.update(filename=filename, shape=shape[:2])
        # if not self.only_resize:
        #     data["is_training"] = True
        # else:
        #     data["is_training"] = False
        data["is_training"] = self.is_training
        return data


class AugmentDetectionData(AugmentData):
    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for line in data["lines"]:
            if self.only_resize:
                new_poly = [(p[0], p[1]) for p in line["poly"]]
            else:
                new_poly = self.may_augment_poly(aug, shape, line["poly"])
            line_polys.append({
                "points": new_poly,
                "ignore": line["text"] == "###",
                "text": line["text"],
            })
        data["polys"] = line_polys
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly
