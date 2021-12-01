import logging
import os
from typing import Any, Dict, List
from urllib.parse import urlparse

import boto3
import mmcv
from botocore.exceptions import ClientError
from label_studio.core.settings.base import DATA_UNDEFINED_NAME
from label_studio.core.utils.io import get_data_dir

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path, get_image_size
from submodule_mmdetection.mmdet.apis import init_detector
from submodule_mmocr.mmocr.apis.inference import model_inference
from submodule_mmocr.mmocr.datasets.pipelines.crop import crop_img

logger = logging.getLogger(__name__)


def det_and_recog_inference(image_path, batch_mode, batch_size, det_model, recog_model):
    end2end_res = {"filename": image_path}
    end2end_res["result"] = []

    image = mmcv.imread(image_path)
    det_result = model_inference(det_model, image)
    bboxes = det_result["boundary_result"]

    box_imgs = []
    for bbox in bboxes:
        box_res = {}
        box_res["box"] = [round(x) for x in bbox[:-1]]
        box_res["box_score"] = float(bbox[-1])
        box = bbox[:8]
        if len(bbox) > 9:
            min_x = min(bbox[0:-1:2])
            min_y = min(bbox[1:-1:2])
            max_x = max(bbox[0:-1:2])
            max_y = max(bbox[1:-1:2])
            box = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
        box_img = crop_img(image, box)
        if batch_mode:
            box_imgs.append(box_img)
        else:
            recog_result = model_inference(recog_model, box_img)
            text = recog_result["text"]
            text_score = recog_result["score"]
            if isinstance(text_score, list):
                text_score = sum(text_score) / max(1, len(text))
            box_res["text"] = text
            box_res["text_score"] = text_score

        end2end_res["result"].append(box_res)

    if batch_mode:
        batch_size = batch_size
        for chunk_idx in range(len(box_imgs) // batch_size + 1):
            start_idx = chunk_idx * batch_size
            end_idx = (chunk_idx + 1) * batch_size
            chunk_box_imgs = box_imgs[start_idx:end_idx]
            if len(chunk_box_imgs) == 0:
                continue
            recog_results = model_inference(
                recog_model, chunk_box_imgs, batch_mode=True)
            for i, recog_result in enumerate(recog_results):
                text = recog_result["text"]
                text_score = recog_result["score"]
                if isinstance(text_score, list):
                    text_score = sum(text_score) / max(1, len(text))
                end2end_res["result"][start_idx + i]["text"] = text
                end2end_res["result"][start_idx + i]["text_score"] = text_score

    return end2end_res


class MMOCR(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection"""

    def __init__(self, det_config, det_ckpt, recog_config, recog_ckpt, image_dir=None, device="cpu", **kwargs):
        """
        Load MMDetection model from config and checkpoint into memory.
        (Check https://mmdetection.readthedocs.io/en/v1.2.0/GETTING_STARTED.html#high-level-apis-for-testing-images)

        Optionally set mappings from COCO classes to target labels
        :param config_file: Absolute path to MMDetection config file (e.g. /home/user/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x.py)
        :param checkpoint_file: Absolute path MMDetection checkpoint file (e.g. /home/user/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth)
        :param image_dir: Directory where images are stored (should be used only in case you use direct file upload into Label Studio instead of URLs)
        :param device: device (cpu, cuda:0, cuda:1, ...)
        :param kwargs:
        """
        super(MMOCR, self).__init__(**kwargs)

        self.det_config = det_config
        self.det_ckpt = det_ckpt
        self.recog_config = recog_config
        self.recog_ckpt = recog_ckpt
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), "media", "upload")
        self.image_dir = image_dir or upload_dir
        logger.debug(
            f"{self.__class__.__name__} reads images from {self.image_dir}")
        print("Load new model")
        self.device = device
        # build detect model
        self.detect_model = init_detector(
            self.det_config, self.det_ckpt, device=self.device)
        if hasattr(self.detect_model, "module"):
            self.detect_model = self.detect_model.module
        if self.detect_model.cfg.data.test["type"] == "ConcatDataset":
            self.detect_model.cfg.data.test.pipeline = \
                self.detect_model.cfg.data.test["datasets"][0].pipeline

        # build recog model
        self.recog_model = init_detector(
            self.recog_config, self.recog_ckpt, device=self.device)
        if hasattr(self.recog_model, "module"):
            self.recog_model = self.recog_model.module
        if self.recog_model.cfg.data.test["type"] == "ConcatDataset":
            self.recog_model.cfg.data.test.pipeline = \
                self.recog_model.cfg.data.test["datasets"][0].pipeline

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

    def single_predict(self, task, **kwargs):
        image_url = self._get_image_url(task)
        image_path = get_image_local_path(image_url, image_dir=self.image_dir)
        model_results = det_and_recog_inference(image_path, batch_mode=True, batch_size=8,
                                                det_model=self.detect_model, recog_model=self.recog_model)["result"]
        results = []
        img_width, img_height = get_image_size(image_path)
        for i, item in enumerate(model_results):
            cell_id = image_url.split("/")[-1] + f"_{i}"
            bbox = item["box"]
            text = item["text"]
            bbox = list(bbox)
            x, y, xmax, ymax = min(bbox[::2]), min(bbox[1::2]), max(bbox[::2]), max(bbox[1::2])
            results += [
                {
                    "id": cell_id,
                    "type": "rectanglelabels",
                    "value": {
                        "x": x / img_width * 100,
                        "y": y / img_height * 100,
                        "width": (xmax - x) / img_width * 100,
                        "height": (ymax - y) / img_height * 100,
                        "rotation": 0,
                        "rectanglelabels": [
                            "Rectangle"
                        ]
                    },
                    "to_name": "image",
                    "from_name": "label",
                    "image_rotation": 0,
                    "original_width": 1121,
                    "original_height": 459
                },
                {
                    "id": cell_id,
                    "type": "textarea",
                    "value": {
                        "x": x / img_width * 100,
                        "y": y / img_height * 100,
                        "width": (xmax - x) / img_width * 100,
                        "height": (ymax - y) / img_height * 100,
                        "text": [
                            text
                        ],
                        "rotation": 0
                    },
                    "to_name": "image",
                    "from_name": "transcription",
                    "image_rotation": 0,
                    "original_width": 1121,
                    "original_height": 459
                }
            ]
        return {"result": results, "score": 0}
