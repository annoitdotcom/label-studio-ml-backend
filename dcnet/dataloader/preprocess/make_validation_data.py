import math

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from dcnet.dataloader.preprocess.preprocess_base import PreprocessBase


class MakeValidationData(PreprocessBase):
    """
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    """
    min_text_size = 4
    shrink_ratio = 0.4

    def __init__(self, opt):
        self.opt = opt.dataset.validation
        self.args = dict()
        self.args["image_resize"] = self.opt.resize_image_shape

    @classmethod
    def load_opt(cls, opt, is_training):
        return cls(opt)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args["image_resize"]
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args["image_resize"]
            new_height = int(math.ceil(new_width / width * height / 32) * 32)

        resized_img = cv2.resize(img, (new_width, new_height))
        ratio = resized_img.shape[0] / height
        return resized_img, ratio

    def preprocess_image(self, img):
        original_shape = img.shape[:2]
        img, ratio = self.resize_image(img)
        return img, ratio

    def __call__(self, data):
        """
        requied keys:
            image, polygons, ignore_tags, filename
        adding keys:
            mask
        """
        image, ratio = self.preprocess_image(data["image"])
        filename = data["filename"]

        polygons = []
        ignore_tags = []
        for line in data["lines"]:
            polygons.append(
                np.array([(int(p[0] / ratio), int(p[1] / ratio)) for p in line["poly"]]))
            if line["text"] == "###":
                ignore_tags.append(True)
            else:
                ignore_tags.append(False)

        ignore_tags = np.array(ignore_tags, dtype=np.uint8)

        h, w = image.shape[:2]
        gt = np.zeros((1, h, w), dtype=np.float32)

        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(polygons)):
            polygon = polygons[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])

            # if ignore_tags[i] or min(height, width) < self.min_text_size:
            if min(height, width) < self.min_text_size:
                # if ignore_tags[i]:
                cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                polygon_shape = Polygon(polygon)
                distance = polygon_shape.area * \
                    (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length

                subject = [tuple(l) for l in polygons[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(mask, polygon.astype(
                        np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt[0], [shrinked.astype(np.int32)], 1)

        if filename is None:
            filename = ""
        data.update(image=image,
                    polygons=polygons,
                    gt=gt, mask=mask, filename=filename)

        return data

    def polygon_area(self, polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * \
                (polygon[next_index, 1] - polygon[i, 1])

        return edge / 2.
