#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Differential contour text-detection network. """

import gc
import logging
import os
import sys
from typing import List

import anyconfig
import cv2
import munch
import numpy as np
import torch
from PIL import Image

from dcnet.inferencer import Inferencer
from dcnet.network_builder import NetworkBuilder
from dcnet.trainer import Trainer
from dcnet.trainer.losses import str2loss
from dcnet.trainer.losses.loss_base import LossBase
from dcnet.utils.input import cast_image_to_array, handle_single_input
from dcnet.utils.seldon_wrapper import Seldon

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class DCNet(Seldon):
    def __init__(self, weights_path: str, config_path: str = None, onnx_path: str = None, use_onnx=False,
                 resize_image_shape: int = 736, box_thres: float = 0.3, thresh: float = 0.3,
                 device: str = "cpu", **kwargs):
        """ Differential contour text-detection network
        Args:
            weights_path: Path to the model weight.
            config_path: Path to the config file.
            resize_image_shape: The new image size to be normalized.
            box_thres: Box threshold.
            thresh: Segmentation mask threshold.
            device: Type of device `cpu` or `cuda`.
        """
        torch.set_num_threads(torch.get_num_threads())
        torch.set_flush_denormal(True)
        self.device = self._init_device(device)
        if config_path:
            self.opt = self._init_config(config_path)
            self.criterion = self._init_criterion(self.opt)
            self.net = NetworkBuilder(
                self.opt,
                self.criterion,
                self.device
            )
            self.trainer = Trainer(self.net, config_path, device=self.device)

        if weights_path:
            self.inferencer = Inferencer(weights_path=weights_path, onnx_path=onnx_path,
                                         use_onnx=use_onnx, box_thresh=box_thres, thresh=thresh,
                                         resize_image_shape=resize_image_shape, device=self.device)

    def _init_config(self, opt_path: str) -> munch.Munch:
        """Initialze all configurations.

        Args:
            opt (dict): dict of all configs
        """
        configs = anyconfig.load(opt_path)
        munch_configs = munch.munchify(configs)

        if munch_configs.optimize_settings.distributed:
            torch.cuda.set_device(munch_configs.optimize_settings.local_rank)
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://"
            )
        torch.backends.cudnn.benchmark = munch_configs.optimize_settings.benchmark
        logging.info("Config setup: {0}".format(munch_configs))
        return munch_configs

    def _init_device(self, device_name: str) -> torch.device:
        if (device_name == "cpu" or device_name == "cuda"):
            device = torch.device(device_name)
        else:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Device chosen: {0}".format(device))
        return device

    def _init_criterion(self, opt) -> LossBase:
        try:
            criterion = str2loss[opt.optimize_settings.loss.type].load_opt(opt)
            logging.debug("Loss type: {0}".format(
                criterion.__class__.__name__))

            criterion = self._parallelize(
                criterion,
                opt.optimize_settings.distributed,
                opt.optimize_settings.local_rank
            )
            return criterion
        except Exception as error:
            logging.error("Error at %s", "division", exc_info=error)

    def _parallelize(self, instance, distributed, local_rank):
        if distributed:
            return torch.nn.parallel.DistributedDataParallel(
                instance,
                device_ids=[local_rank],
                output_device=[local_rank],
                find_unused_parameters=True
            )
        else:
            return torch.nn.DataParallel(instance)

    def _rgb_to_gray(self, image: np.array) -> np.array:
        """Convert rgb image to gray scale
        Args:
            image (np.array): input a numpy array image, shape == 3
        Return:
            a converted numpy array image, shape == 2
        """
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image

    def _preprocess(self, image: np.array) -> np.array:
        """Process input image to numpy array type with grayscale (dim == 2)
        Args:
            image (str, np.array, PIL.Image): input image
        Return:
            image (np.array): grayscale numpy array image
        """
        if isinstance(image, str):
            np_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image, np.ndarray):
            np_image = self._rgb_to_gray(image)
        elif isinstance(image, Image.Image):
            np_image = self._rgb_to_gray(np.array(image))
        else:
            raise ValueError(
                "Only support for str, np.array and PIL, but found {0}".format(type(image)))
        return np_image

    def train(self):
        output = self.trainer.train()
        return output

    def save(self, weights_path: str) -> None:
        self.trainer.save(weights_path)

    def load(self, weights_path=None):
        self.trainer.load(weights_path)

    def convert_to_onnx(self, onnx_path: str) -> str:
        onnx_path = self.inferencer.convert_to_onnx(onnx_path)
        return onnx_path

    @handle_single_input(cast_image_to_array)
    def process(self, images: List[np.array], get_mask: bool = False, pred_mask=None, inference_resize=896):
        """ Predict images, support list of string, list of numpy array, path string, np.array, PIL.Image
        Args:
            images : a single `str`, `PIL.Image`, `numpy.ndarray`
                    or an iterable containing them
            get_mask: Whether to return segmentation mask or not.
            pred_mask: Annotated segmentation mask.
            inference_resize: Size to normalize the input image.
        Returns:
            List of dicts with keys `location`, `text type`.
        """
        if type(images) == list and not images:
            return []

        np_images = list(map(self._preprocess, images))
        n_images = len(np_images)
        results = []
        with torch.no_grad():
            for inp_img in np_images:
                output = self.inferencer.predict(
                    inp_img, get_mask, pred_mask, inference_resize)
                results.append(output)
            assert len(
                results) == n_images, "[ERROR] Mismatch length of output and input"
            gc.collect()
            return results
