import cv2
import numpy as np
import pyclipper
import torch
from shapely.geometry import Polygon

from dcnet.inferencer.auxiliary.utils import (get_min_max_xy, preprocess_image,
                                              resize_image)


class DcPostprocessor:

    def __init__(self, args):
        self.min_size = 3
        self.scale_ratio = 0.4
        self.max_candidates = 100000
        self.dest = "binary"
        self.debug = False

        self.args = args
        self.thresh = self.args["thresh"]
        self.box_thresh = self.args["box_thresh"]

    def postprocess(self, batch, batch_boxes, batch_scores):
        list_boxes = []
        pad_width = 5
        pad_height = 5

        for index in range(batch["image"].size(0)):
            original_shape = batch["shape"][index]
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args["polygon"]:
                for i, box in enumerate(boxes):
                    box = np.array(box).reshape(-1).tolist()
                    all_xy = [int(x) for x in box]
                    (minx, miny, maxx, maxy) = get_min_max_xy(all_xy)
                    list_boxes.append({"location": [[minx, miny], [maxx, miny], [
                                      maxx, maxy], [minx, maxy]], "type": "textline"})
            else:
                for i in range(boxes.shape[0]):
                    score = scores[i]
                    if score < self.args["box_thresh"]:
                        continue
                    box = boxes[i, :, :].reshape(-1).tolist()
                    all_xy = [int(x) for x in box]

                    (minx, miny, maxx, maxy) = get_min_max_xy(all_xy)
                    box_loc = []
                    for ii in range(int(len(all_xy) / 2)):
                        box_loc.append([all_xy[ii * 2], all_xy[ii * 2 + 1]])
                    list_boxes.append(
                        {"location": box_loc, "type": "textline"})

        return list_boxes

    def process(self, batch, _pred, is_output_polygon=False):
        """
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        pred:
            binary: text region segmentation map, with shape (N, 1, H, W)
            thresh: [if exists] thresh hold prediction with shape (N, 1, H, W)
            thresh_binary: [if exists] binarized with threshhold, (N, 1, H, W)
        """
        images = batch["image"]

        if isinstance(_pred, dict):
            pred = _pred[self.dest]
        else:
            pred = _pred
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        for batch_index in range(images.size(0)):
            height, width = batch["shape"][batch_index]
            if is_output_polygon:
                boxes, scores = self.polygons_from_bitmap(
                    pred[batch_index],
                    segmentation[batch_index], width, height)
            else:
                boxes, scores = self.boxes_from_bitmap(
                    pred[batch_index],
                    segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)

        result = self.postprocess(batch, boxes_batch, scores_batch)
        return result

    def binarize(self, pred):
        # cv2.imwrite("demo_mask_pred.jpg", pred.cpu().numpy()[0][0] * 255)
        # cv2.imwrite("demo_mask_binaried.jpg", (pred > self.thresh).cpu().numpy()[0][0] * 255)
        binarized = (pred > self.thresh).cpu().numpy()[0] * 255
        _, h, w = binarized.shape
        binarized = np.reshape(binarized, (h, w, 1)).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)

        img_dilation = cv2.dilate(binarized, kernel, iterations=1)
        # cv2.imwrite("demo_mask_dilation.jpg", img_dilation)

        kernel = np.ones((2, 2), np.uint8)

        img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
        # cv2.imwrite("demo_mask_erosion.jpg", img_erosion)

        # return torch.from_numpy(img_erosion / 255).reshape((1, 1, h , w))
        return pred > self.thresh

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        """

        assert _bitmap.size(0) == 1
        bitmap = _bitmap.cpu().numpy()[0]  # The first channel
        pred = pred.cpu().detach().numpy()[0]
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap*255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.max_candidates = min(self.max_candidates, len(contours))
        for contour in contours[:self.max_candidates]:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=2.0)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        """

        assert _bitmap.size(0) == 1

        bitmap = _bitmap.cpu().numpy()[0]  # The first channel
        pred = pred.cpu().detach().numpy()[0]
        height, width = bitmap.shape
        try:
            contours, _ = cv2.findContours(
                (bitmap*255).astype(np.uint8),
                cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        except Exception as e:
            _, contours, _ = cv2.findContours(
                (bitmap*255).astype(np.uint8),
                cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if score < self.box_thresh:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)

        return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]
