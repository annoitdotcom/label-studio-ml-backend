import os
import logging
import boto3
import io

from mmdet.apis import init_detector, inference_detector

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path, get_image_size, get_single_tag_keys
from label_studio.core.utils.io import json_load, get_data_dir
from label_studio.core.settings.base import DATA_UNDEFINED_NAME
from botocore.exceptions import ClientError
from urllib.parse import urlparse
from google.cloud import vision
import numpy as np
from enum import Enum


class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5
    
document_bkp = None

logger = logging.getLogger(__name__)


def get_bbox(vertices):
    return [min([v.x for v in vertices]), min([v.y for v in vertices]),
            max([v.x for v in vertices]), max([v.y for v in vertices])]


def get_polygons(vertices):
    indexes = list(range(len(vertices)))
    indexes = [(i, i +1) for i in indexes[::2]][::2] + [(i, i +1) for i in indexes[2::2]][::2][::-1]
    indexes = [item for t in indexes for item in t]
    return [(vertices[i].x, vertices[i].y) for i in indexes]


def full_doc_ocr(image_file):
#     global document_bkp
    
#     if document_bkp is not None:
#         print("BKP")
#         document = document_bkp
    
#     else:

    client_options = {"api_endpoint": "eu-vision.googleapis.com"}

    client = vision.ImageAnnotatorClient(client_options=client_options)

    with io.open(image_file, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image, image_context={"language_hints": ["ja", "en"]})
    document = response.full_text_annotation

#     document_bkp = document

    paragraph_list = []

    breaks = vision.TextAnnotation.DetectedBreak.BreakType
    
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                paragraph_dct = {
                    "text": "",
                    "bbox": get_bbox(paragraph.bounding_box.vertices),
                    "lines": []
                }
                
                line_text = ""
                line_vs = []
                line_words = []
                break_type = None
                
                for word in paragraph.words:
                    words_dct = {
                        "text": "",
                        "bbox": get_bbox(word.bounding_box.vertices),
                        "symbols": []
                    }
                    
                    for symbol in word.symbols:
                        symbol_dct = {
                            "text": symbol.text,
                            "confidence": symbol.confidence,
                            "bbox": get_bbox(symbol.bounding_box.vertices)
                        }
                        
                        words_dct["symbols"].append(symbol_dct)
                        words_dct["text"] += symbol.text
                        
                        line_text += symbol.text
                        line_vs += list(symbol.bounding_box.vertices)
                        
                        line_break = str(symbol.property.detected_break)
                        break_type = line_break.split(": ")[1].strip() if len(line_break) else None
                        
                        if break_type == "SPACE":
                            line_text += " "

                        elif break_type in ["LINE_BREAK", "EOL_SURE_SPACE"]:
                            line_dct = {
                                "text": line_text,
                                "bbox": get_bbox(line_vs),
                                "polygon": get_polygons(line_vs),
                                "words": line_words + [words_dct]
                            }
                            
                            paragraph_dct["lines"].append(line_dct)
                            paragraph_dct["text"] = paragraph_dct["text"] + line_text + "\n"
                            
                            line_text = ""
                            line_vs = []
                            line_words = []
                    
                    line_words.append(words_dct)
                
                paragraph_dct["text"] = paragraph_dct["text"][:-1]
                paragraph_list.append(paragraph_dct)

    
    for paragraph in paragraph_list:
        paragraph["confidence"] = 1.0
        
        for line in paragraph["lines"]:
            line["confidence"] = 1.0
            
            for word in line["words"]:
                word["confidence"] = 1.0
                
                for symbol in word["symbols"]:
                    word["confidence"] *= symbol["confidence"]
                    
                line["confidence"] *= word["confidence"]
                
            paragraph["confidence"] *= line["confidence"]

    return paragraph_list


def expand_paragraph_list(paragraph_list):
    paragraphs = []
    lines = []
    words = []
    symbols = []
    
    for paragraph in paragraph_list:
        
        for line in paragraph["lines"]:
            
            for word in line["words"]:
                
                for symbol in word["symbols"]:
                    symbols.append([symbol["bbox"], symbol["text"], symbol["confidence"]])
                    
                words.append([word["bbox"], word["text"], word["confidence"]])
            
            lines.append([line["bbox"], line["polygon"], line["text"], line["confidence"]])
        
        paragraphs.append([paragraph["bbox"], paragraph["text"], paragraph["confidence"]])

    return paragraphs, lines, words, symbols


def simple_ocr(image_file):
    
    client_options = {"api_endpoint": "eu-vision.googleapis.com"}
    
    client = vision.ImageAnnotatorClient(client_options=client_options)

    with io.open(image_file, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image, image_context={"language_hints": ["ja", "en"]})
    document = response.text_annotations

    results = []
    
    for paragraph in document:
        text = paragraph.description
        vs = paragraph.bounding_poly.vertices
        box = [min([v.x for v in vs]), min([v.y for v in vs]), max([v.x for v in vs]), max([v.y for v in vs])]
        
        results.append([box, text, 1.0])
        
    return results


def draw_res(img_pil, res):
    font = PIL.ImageFont.truetype("/home/ec2-user/matheus/notebooks/simsun.ttc", 32)
    draw = PIL.ImageDraw.Draw(img_pil)

    for boxes, polygons, text, conf in res:
        x1, y1, x2, y2 = boxes
        
#         draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0))
        draw.polygon(polygons, outline ="blue")
#         draw.text((x1, y1-20), text, font=font, fill=(255, 0, 0), stroke_width=1)

    return img_pil


class GoogleOCR(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection"""

    def __init__(self, credentials_path, image_dir=None, **kwargs):
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
        super(GoogleOCR, self).__init__(**kwargs)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), "media", "upload")
        self.image_dir = image_dir or upload_dir
        logger.debug(f"{self.__class__.__name__} reads images from {self.image_dir}")

    def _get_image_url(self, task):
        image_url = task["data"].get("ocr") or task["data"].get(DATA_UNDEFINED_NAME)
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
                logger.warning(f"Can\"t generate presigned URL for {image_url}. Reason: {exc}")
        return image_url

    def predict(self, tasks, **kwargs):
        assert len(tasks) == 1
        task = tasks[0]
        if len(task.get("predictions", [])) > 0:
            return task["predictions"]
        image_url = self._get_image_url(task)
        image_path = get_image_local_path(image_url, image_dir=self.image_dir)

        paragraph_list = full_doc_ocr(image_path)

        paragraphs, lines, words, symbols = expand_paragraph_list(paragraph_list)

        # polygons and texts in `lines` variable
        results = []
        img_width, img_height = get_image_size(image_path)

        for i, (boxes, polygons, text, conf) in enumerate(lines):
            cell_id = image_url.split("/")[-1] + f"_{i}"

            results += [
                {
                    "id": cell_id,
                    "type": "polygonlabels",
                    "value": {
                        "points": [[p[0] / img_width * 100, p[1] / img_height * 100] for p in polygons],
                        "polygonlabels": [
                            "other"
                        ]
                    },
                    "to_name": "image",
                    "from_name": "label",
                    "image_rotation": 0,
                    "original_width": img_width,
                    "original_height": img_height
                },
                {
                    "id": cell_id,
                    "type": "textarea",
                    "value": {
                        "points": [[p[0] / img_width * 100, p[1] / img_height * 100] for p in polygons],
                        "text": [
                            text
                        ],
                    },
                    "to_name": "image",
                    "from_name": "transcription",
                    "image_rotation": 0,
                    "original_width": img_width,
                    "original_height": img_height
                }
            ]
        # import pdb; pdb.set_trace()
        # print(
        #     [{
        #     "result": results,
        #     "score": 0
        # }]
        # )
        predictions = [{"result": results, "score": 0}]
        return predictions
