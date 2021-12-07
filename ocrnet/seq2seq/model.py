#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" An light-weight, robust OCR model based on 
sequence to sequence modeling, aiming to 
fasten the processing speed. 
"""

import gc
import logging

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

from ocrnet.seq2seq.core.model import SequenceModel
from ocrnet.seq2seq.utils.tools import (AlignCollate, CTCLabelConverter,
                                        count_parameters)
from ocrnet.utils.input import cast_image_to_array, handle_single_input
from ocrnet.utils.seldon_wrapper import Seldon

SUPPORT_MODES = ['inference']  # Current support modes
logging.basicConfig(level=logging.INFO)


class Seq2SeqOCR(Seldon):
    """ Sequence to sequence modeling based OCR. """

    def __init__(self, weights_path=None, device=None, batch_size=4, mode="inference", **kwargs):
        torch.set_num_threads(torch.get_num_threads())
        torch.set_flush_denormal(True)

        if (device == 'cpu' or device == 'cuda'):
            self.device = torch.device(device)
        elif device == '-1':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda'
                                       if torch.cuda.is_available() else 'cpu')
        logging.info("device chosen: {0}".format(self.device))

        if mode == "inference":
            self.model, self.model_config, \
                self.alphabet = self._load_model(weights_path)
        else:
            raise ValueError(
                "Current only support for {0} mode, but found {1}".format(SUPPORT_MODES, mode))

        # Set model to eval mode.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)

        # Pre-process input images
        self.input_transform = AlignCollate(
            imgH=self.model_config.img_height,
            keep_ratio_with_pad=True)
        self.batch_size = batch_size
        self.converter = CTCLabelConverter(self.alphabet)

    def _load_model(self, weights_path):
        """ Model configuration 
        Args: 
            weights_path (str): input model weight path
        Return:
            model: loaded model
            model_config (dict): dict of model configure
            alphabet (lst): list of trained characters
        """
        logging.info("loading pretrained model from {0}".format(weights_path))
        checkpoint = torch.load(weights_path,
                                map_location=lambda storage, loc: storage)
        model_state = checkpoint.get("state_dict", None)
        model_config = checkpoint.get("configs", None)
        alphabet = checkpoint.get("vocabulary", None)

        model = SequenceModel(model_config)
        logging.info("Model number params: {0}".format(
            count_parameters(model)))
        try:
            model.load_state_dict(model_state)
        except RuntimeError:
            logging.info(
                "Could not load_state_dict, retrying with DataParallel loading mode...")
            model = torch.nn.DataParallel(model)
            model.load_state_dict(model_state)
            logging.info("Loading Success!")
        return (model, model_config, alphabet)

    def _rgb_to_gray(self, image):
        """ convert rgb image to gray scale
        Args:
            image (np.array): input a numpy array image, shape == 3
        Return:
            a converted numpy array image, shape == 2
        """
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image

    def _preprocess(self, image):
        """ process input image to numpy array type with grayscale (dim == 2)
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

    def _predict(self, input_tensor, get_logit=False):
        """ Predict for single image
        Args:
            input_tensor (Torch.Tensor): input image that have been processed 
                to torch tensor with dim (B x C x H x W) == (1, 1, H, W)
            get_logit (boolean): True is return with output logits, \
                else False (mainly use for confidence tuning)
        Return:
            output (dict): outputs a dict contains:
                predicted text (str)
                confidence by character (lst): a list of confidence score for each character
                confidence by field (float): confidence by field
        """
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            # input_var = torch.autograd.Variable(input_tensor, requires_grad=False)
            preds = self.model(input_tensor).log_softmax(2)
            # preds = preds.detach()

            # Select max probabilty (greedy decoding) then decode index to character
            preds_str, confidence_scores = self.converter.decode(preds)
            confidence_by_character = confidence_scores
            confidence_by_field = list(
                map(lambda s: np.prod(s), confidence_scores))

            if get_logit == True:
                output = list(map(lambda k: {
                    "text": k[0],
                    "confidence_by_character": list(map(lambda s: round(s, 4), k[1])),
                    "confidence_by_field": round(k[2], 4),
                    "yp": k[3].squeeze().cpu().data.numpy()
                }, zip(preds_str, confidence_by_character, confidence_by_field, preds)))
            else:
                output = list(map(lambda k: {
                    "text": k[0],
                    "confidence_by_character": list(map(lambda s: round(s, 4), k[1])),
                    "confidence_by_field": round(k[2], 4)
                }, zip(preds_str, confidence_by_character, confidence_by_field)))
            return output

    @handle_single_input(cast_image_to_array)
    def process(self, images, batch_size=None, get_logit=False, sort_by_width=True):
        """ Predict images
        Args:
            images : a single `str`, `PIL.Image`, `numpy.ndarray`
                    or an iterable containing them
            batch_size : temporal batch_size only for one-time process call
                    when set with integer, it will over-rule the default batch_size 
                    of object's __init__ when run the process
            sort_by_width <bool>: sort input images by width before processing
                    (recommend for large list of input images with various widths,
                    however, consider turn this off to save sorting time when 
                    input is only a few images)
        Returns:
            final_outputs : list of dicts with keys `text`, `confidence_by_character`, `confidence_by_field`
        """
        if type(images) == list and not images:
            return []
        if batch_size is not None and isinstance(batch_size, int):
            batch_size = batch_size
        else:
            batch_size = self.batch_size

        np_images = list(map(self._preprocess, images))
        n_images = len(np_images)
        n_batchs = (n_images - 1) // batch_size + 1

        # Sort images by width
        if sort_by_width:
            sorted_pairs = sorted(
                enumerate(np_images), key=lambda x: x[1].shape[1])  # sort with width
            sorted_idxs = [x[0] for x in sorted_pairs]
            sorted_images = [x[1] for x in sorted_pairs]
        else:
            sorted_images = np_images  # Not sort input

        results = []
        with torch.no_grad():
            for x in range(n_batchs):
                batch_images = sorted_images[x*batch_size:(x+1)*batch_size]
                tensor_images = self.input_transform(batch_images)
                batch_results = self._predict(
                    tensor_images, get_logit)  # Run predict
                results.extend(batch_results)
            assert len(
                results) == n_images, "[ERROR] Mismatch length of output and input"

            # Re-sort results to match original order of input
            if sort_by_width:
                results_order = sorted(
                    enumerate(results), key=lambda x: sorted_idxs[x[0]])
                final_outputs = [x[1] for x in results_order]
            else:
                final_outputs = results
            gc.collect()
            return final_outputs
