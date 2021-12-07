import errno
import json
import os
from typing import List

import numpy as np
from shapely.geometry import Polygon


def _get_polygon_iou(gt_box: List[tuple], pd_box: List[tuple]) -> int:
    """Intersection over union on layout polygon

        Parameters
        -------------
        gt_box: List[tuple]
            A list contains bounding box coordinates of ground truth
        pd_box: List[tuple]
            A list contains bounding box coordinates of prediction
    """
    # get polygon of gd_box and pd_box
    gt_polygon = Polygon(gt_box)
    pd_polygon = Polygon(pd_box)

    # calculate intersection and union area
    try:
        intersection = gt_polygon.intersection(pd_polygon).area
        union = gt_polygon.union(pd_polygon).area
    except:
        print('The geometry of bounding box is not valid')
        return 0

    return intersection / union


def _get_rect_iou(gt_box: List[tuple], pd_box: List[tuple]) -> int:
    """Intersection over union on layout rectangle

    Parameters
    -------------
    gt_box: List[tuple]
        A list contains bounding box coordinates of ground truth
    pd_box: List[tuple]
        A list contains bounding box coordinates of prediction
    """

    # determine the (x, y)-coordinates of the intersection rectangle
    #gt_box: [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    #pd_box: [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    x_left = max(gt_box[0][0], pd_box[0][0])
    y_top = max(gt_box[0][1], pd_box[0][1])
    x_right = min(gt_box[2][0], pd_box[2][0])
    y_bottom = min(gt_box[2][1], pd_box[2][1])

    # compute the area of intersection rectangle
    interArea = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # compute the area of both the prediction and ground-truth
    # rectangles
    gt_area = (gt_box[2][0] - gt_box[0][0]) * \
        (gt_box[2][1] - gt_box[0][1])

    pd_area = (pd_box[2][0] - pd_box[0][0]) * \
        (pd_box[2][1] - pd_box[0][1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(gt_area + pd_area - interArea)

    # return the intersection over union value
    return iou


def _parse_from_io_dict(boxes: List[dict] or List[list] or List[tuple]) -> List[tuple]:
    """Function to parse all bounding boxes from lib_layout io-specification standard output format

    Parameters
    -------------
    box: List[dict]
        List contains all dictionaries in compliance with io-specification format which contains bounding box information
    """

    all_boxes = []
    for text_line_region in boxes:
        box = dict()
        if isinstance(text_line_region, dict):
            if len(text_line_region['location']) == 4:
                box['rect'] = text_line_region['location']
            else:
                box['polygon'] = text_line_region['location']
        else:
            if len(text_line_region) == 4:
                box['rect'] = text_line_region
            else:
                box['polygon'] = text_line_region

        all_boxes.append(box)
    return all_boxes


def _parse_from_json(json_path: str) -> dict:
    """Function to parse all bounding boxes from json file

    Parameters
    -------------
    json_path: str
        The path of json file that contains information of bounding box
    """
    # error handling
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), json_path)

    all_boxes = []

    # read json file
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # handle DataPile type of label
    if isinstance(json_data, dict):
        # parse bounding box attribute from json file
        text_line_regions = json_data['attributes']['_via_img_metadata']['regions']
        for region in text_line_regions:
            box = dict()
            shape_attr = region['shape_attributes']

            # handling `rect` label
            if shape_attr['name'] == 'rect':
                x1, y1 = shape_attr['x'], shape_attr['y']
                x2, y2 = shape_attr['x'] + \
                    shape_attr['width'], shape_attr['y'] + shape_attr['height']
                box['rect'] = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                #all_boxes.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

            # handling `polygon` label
            elif shape_attr['name'] == 'polygon':
                xs = shape_attr['all_points_x']
                ys = shape_attr['all_points_y']
                box['polygon'] = [point for point in zip(xs, ys)]
                #all_boxes.append([point for point in zip(xs, ys)])

            all_boxes.append(box)

    # handle lib_layout io-specification type of label
    # ex: ({'location': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], 'type': 'textline'})
    elif isinstance(json_data, list):
        for text_line_region in json_data:
            box = dict()
            if len(text_line_region['location']) == 4:
                box['rect'] = text_line_region['location']

            else:
                box['polygon'] = text_line_region['location']

            all_boxes.append(box)

    else:
        raise TypeError

    return all_boxes


def _get_TP(gt_boxes: List[dict], pd_boxes: List[List], threshold: float = 0.5) -> int:
    """Function to get number of truth positive samples

    Parameters
    -------------
    gt_boxes: dict
        dict of list contains all bounding box coordinates of all ground truth text lines
        `[{'rect': [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]}, ...]`
    pd_boxes: List
        List contains all bounding box coordinates of all predicted text lines
        `[[(x1, y1), (x2, y1), (x2, y2), (x1, y2)], [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], ..]
    threshold: int
        threshold to verify the bounding box is TP or not, default is 0.5
    """
    tp = 0
    for pd_box in pd_boxes:
        # parse out the bounding box
        pd_box = list(pd_box.values())[0]
        cnt = 0  # use to handle if the predicted bounding box is too big
        for gt_box in gt_boxes:
            # calculate iou for `rectangular` box
            if 'rect' in gt_box:
                iou = _get_rect_iou(gt_box['rect'], pd_box)
            # calculate iou for `polygon` box
            elif 'polygon' in gt_box:
                iou = _get_polygon_iou(gt_box['polygon'], pd_box)

            if iou >= threshold:
                cnt += 1
                if cnt > 1:
                    break

        if cnt == 1:
            tp += 1

    return tp


def _get_precision(pd_boxes: List or int, num_tp: int) -> int:
    """Function to get precision of the model

    Parameter
    ------------
    pd_boxes: List or int
        prediction boxes, could be a list of all boxes contained or number of all prediction boxes
    num_tp: int
        number of true positive samples
    """

    if isinstance(pd_boxes, int):
        precision = num_tp / pd_boxes
    elif isinstance(pd_boxes, List):
        precision = num_tp / len(pd_boxes)
    else:
        raise TypeError

    return precision


def _get_recall(gt_boxes: List or int, num_tp: int) -> int:
    """Function to get recall of the model

    Parameter
    ------------
    gt_boxes: List or int
        ground truth boxes, could be a list of all boxes contained or number of all ground truth boxes
    num_tp: int
        number of true positive samples
    """

    if isinstance(gt_boxes, int):
        recall = num_tp / gt_boxes
    elif isinstance(gt_boxes, List):
        recall = num_tp / len(gt_boxes)
    else:
        raise TypeError

    return recall


def cal_all_metrics(gt_boxes: List or str, pd_boxes: List or List[dict], threshold: float = 0.5) -> List[int]:
    """Function to get f1 score, precision, recall, TP, FP, and FN for the model

    Parameter
    ------------
    gt_boxes: List or str
        ground truth boxes, could be a list of coodinate tuple or a path of json file
    pd_boxes: List
        prediction boxes
    threshold: int
        threshold to verify the bounding box is TP or not, default is 0.5

    Return
    --------
    f1 score: int
    precision: float
    recall: float
    tp: float
    fp: float
    fn: float
    """
    if isinstance(gt_boxes, str):
        # parse from label json file
        gt_boxes = _parse_from_json(gt_boxes)

    else:
        # convert data from io-specification format to desired format
        gt_boxes = _parse_from_io_dict(gt_boxes)

    pd_boxes = _parse_from_io_dict(pd_boxes)
    # get true positive, false positive, and false negative
    tp = _get_TP(gt_boxes, pd_boxes, threshold=threshold)
    fp = len(pd_boxes) - tp
    fn = len(gt_boxes) - tp

    # handling the special case of tp = 0, fp = 0, and fn = 0
    if tp == 0 and fp == 0 and fn == 0:
        precision = 1
        recall = 1
        f1 = 1
        return [f1, precision, recall, tp, fp, fn]
    else:
        precision = _get_precision(pd_boxes, tp)
        recall = _get_recall(gt_boxes, tp)

    # handling 0 truth positive samples
    if tp == 0:
        f1 = 0
    else:
        f1 = 2 * ((recall * precision) / (recall + precision))

    return [f1, precision, recall, tp, fp, fn]
