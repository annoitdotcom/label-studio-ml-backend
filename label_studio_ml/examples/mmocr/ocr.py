import logging
import os
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import boto3
import cv2
import mmcv
import numpy as np
from botocore.exceptions import ClientError
from deskew_image import cv_deskew_image, fft_deskew_image, rotate_image
from label_studio.core.settings.base import DATA_UNDEFINED_NAME
from label_studio.core.utils.io import get_data_dir

from dcnet.model import DCNet
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path, get_image_size
from ocrnet.seq2seq.model import Seq2SeqOCR

logger = logging.getLogger(__name__)


class AnnoitOCR(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection"""

    def __init__(self, layout_model_path, ocr_model_path, image_dir=None, batch_size: int = 16, device="cpu", **kwargs):
        """
        Load MMDetection model from config and checkpoint into memory.
        (Check https://mmdetection.readthedocs.io/en/v1.2.0/GETTING_STARTED.html#high-level-apis-for-testing-images)

        Optionally set mappings from COCO classes to target labels.
        :param layout_model_path: Path layout model checkpoint file (e.g. ./weights/dclayout_model.pt).
        :param ocr_model_path: Path ocr model checkpoint file (e.g. ./weights/ocr_model.pt).
        :param image_dir: Directory where images are stored (should be used only in case you use direct file upload into Label Studio instead of URLs).
        :param batch_size: OCR inferencing batch size.
        :param device: device (cpu, cuda:0, cuda:1, ...).
        :param kwargs:
        """
        super(AnnoitOCR, self).__init__(**kwargs)
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), "media", "upload")
        self.image_dir = image_dir or upload_dir
        logger.debug(
            f"{self.__class__.__name__} reads images from {self.image_dir}")
        logger.debug("Load new model")
        self.device = device
        self.layout_model = DCNet(weights_path=layout_model_path, thresh=0.3, box_thres=0.4, device=self.device)
        self.ocr_model = Seq2SeqOCR(weights_path=ocr_model_path, batch_size=batch_size, device=self.device)

    def _get_image_url(self, task):
        image_url = task["data"].get(
            "ocr") or task["data"].get(DATA_UNDEFINED_NAME)
        if image_url.startswith("s3://"):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip("/")
            client = boto3.client("s3")
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": bucket_name, "Key": key}
                )
            except ClientError as exc:
                logger.warning(
                    f"Can\"t generate presigned URL for {image_url}. Reason: {exc}")
        return image_url

    def predict(self, tasks, **kwargs):
        outputs: List[Dict[str, Any]] = []
        for task in tasks:
            resutls = self.single_predict(task, **kwargs)
            outputs.append(resutls)
        return outputs

    def crop_line(self, image: np.ndarray, rectangle: Tuple[int, int, int, int], padw: int = 0, padh: int = 0) -> np.ndarray:
        """Crops bounding box region from image."""
        image_shape = image.shape
        return image[max(rectangle[1] - padh, 0):min(rectangle[3] + padh, image_shape[0]),
                     max(rectangle[0] - padw, 0):min(rectangle[2] + padw, image_shape[1])]

    def single_predict(self, task, **kwargs):
        image_url = self._get_image_url(task)
        image_path = get_image_local_path(image_url, image_dir=self.image_dir)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        skewed_angle = fft_deskew_image(image)
        if skewed_angle > kwargs.get("max_skew_angle", 30):
            skewed_angle = 0
        image = rotate_image(image, skewed_angle)
        la_outputs = self.layout_model.process(image)

        # Get list of text line images and box entities.
        entity_list = []
        for item in la_outputs:
            location = np.array(item["location"])
            x1 = location[:, 0].min()
            y1 = location[:, 1].min()
            x2 = location[:, 0].max()
            y2 = location[:, 1].max()
            rectangle = [x1, y1, x2, y2]

            # Cropping text box in the image with a padding of 1 to make sure that
            # we would not over-crop in other region which could introduce noise in a cropped image.
            # Since this is a textline CNN-LSTM ocr model, which mainly supports to recognize characters
            # in a horizontal line image. For a vertical line image would cause a problem in convolution
            # kernel to process so therefore we put a constraint that the image height is always smaller
            # than 3 times the line image width.
            img_line = self.crop_line(image, rectangle)
            if img_line.shape[0] <= 3 * img_line.shape[1]:
                entity_list.append([img_line, rectangle])

        # Get image list and box entity list.
        image_list, box_list = zip(*entity_list)

        # Run the ocr prediction with batch, and output a list of dictionaries, each containing prediction items:
        # [{"text": "abc", "confidence_by_character": [0.9, 0.9, 0.9], "confidence_by_field": 0.9}, ...].
        ocr_outputs = self.ocr_model.process(image_list)

        # Format ocr results to expected output format.
        results: List[Dict[str, Any]] = []
        scores: List[float] = []
        img_width, img_height = get_image_size(image_path)
        for idx, (item, bbox) in enumerate(zip(ocr_outputs, box_list)):
            cell_id = image_url.split("/")[-1] + f"_{idx}"
            scores.append(item.get("confidence_by_field", 0.0))
            results += [
                {
                    "id": cell_id,
                    "type": "rectanglelabels",
                    "value": {
                        "x": bbox[0] / img_width * 100,
                        "y": bbox[1] / img_height * 100,
                        "width": (bbox[2] - bbox[0]) / img_width * 100,
                        "height": (bbox[3] - bbox[1]) / img_height * 100,
                        "rotation": 0,
                        "rectanglelabels": [
                            "Rectangle"
                        ]
                    },
                    "to_name": "image",
                    "from_name": "label",
                    "image_rotation": skewed_angle,
                    "original_width": img_width,
                    "original_height": img_height
                },
                {
                    "id": cell_id,
                    "type": "textarea",
                    "value": {
                        "x": bbox[0] / img_width * 100,
                        "y": bbox[1] / img_height * 100,
                        "width": (bbox[2] - bbox[0]) / img_width * 100,
                        "height": (bbox[3] - bbox[1]) / img_height * 100,
                        "text": [
                            item["text"]
                        ],
                        "rotation": skewed_angle
                    },
                    "to_name": "image",
                    "from_name": "transcription",
                    "image_rotation": skewed_angle,
                    "original_width": img_width,
                    "original_height": img_height
                }
            ]
        return {"result": results, "score": np.mean(scores)}
