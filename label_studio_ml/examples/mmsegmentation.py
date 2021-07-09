import os
import logging
import boto3

from mmseg.apis import inference_segmentor, init_segmentor

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path, get_image_size, get_single_tag_keys
from label_studio.core.utils.io import json_load, get_data_dir
from label_studio.core.settings.base import DATA_UNDEFINED_NAME
from botocore.exceptions import ClientError
from urllib.parse import urlparse
from imantics import Polygons, Mask


logger = logging.getLogger(__name__)


class MMSegmentation(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection"""

    def __init__(self, config_file, checkpoint_file, image_dir=None, labels_file=None, score_threshold=0.3, device='cpu', **kwargs):
        """
        Load MMDetection model from config and checkpoint into memory.
        (Check https://mmdetection.readthedocs.io/en/v1.2.0/GETTING_STARTED.html#high-level-apis-for-testing-images)

        Optionally set mappings from COCO classes to target labels
        :param config_file: Absolute path to MMDetection config file (e.g. /home/user/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x.py)
        :param checkpoint_file: Absolute path MMDetection checkpoint file (e.g. /home/user/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth)
        :param image_dir: Directory where images are stored (should be used only in case you use direct file upload into Label Studio instead of URLs)
        :param labels_file: file with mappings from COCO labels to custom labels {"airplane": "Boeing"}
        :param score_threshold: score threshold to wipe out noisy results
        :param device: device (cpu, cuda:0, cuda:1, ...)
        :param kwargs:
        """
        super(MMSegmentation, self).__init__(**kwargs)

        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.labels_file = labels_file
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(f'{self.__class__.__name__} reads images from {self.image_dir}')
        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'PolygonLabels', 'Image')
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)

        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name

        print('Load new model from: ', config_file, checkpoint_file)
        self.model = init_segmentor(config_file, checkpoint_file, device=device)

    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as exc:
                logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        return image_url

    def predict(self, tasks, **kwargs):
        assert len(tasks) == 1
        task = tasks[0]
        image_url = self._get_image_url(task)
        image_path = get_image_local_path(image_url, image_dir=self.image_dir)
        model_results = inference_segmentor(self.model, image_path)
        results = []
        all_scores = []
        img_width, img_height = get_image_size(image_path)
        # TODO: model results is a segmentation map, need to convert to onehot?
        # import pdb; pdb.set_trace()
        for i, c in enumerate(self.model.CLASSES):
            points_list = Mask(model_results[0] == (i + 1)).polygons().points
            output_label = self.label_map.get(c, c)

            if output_label not in self.labels_in_config:
                print(output_label + ' label not found in project config.')
                continue
            for points in points_list:
                points[:, 0] = points[:, 0] / img_width * 100
                points[:, 1] = points[:, 1] / img_height * 100
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'polygonlabels',
                    'value': {
                        'polygonlabels': [output_label],
                        'points': points.tolist()
                    },
                    'score': 0
                })
        return [{
            'result': results,
            'score': 0
        }]
import os
import logging
import boto3

from mmseg.apis import inference_segmentor, init_segmentor

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path, get_image_size, get_single_tag_keys
from label_studio.core.utils.io import json_load, get_data_dir
from label_studio.core.settings.base import DATA_UNDEFINED_NAME
from botocore.exceptions import ClientError
from urllib.parse import urlparse
from imantics import Polygons, Mask


logger = logging.getLogger(__name__)


class MMSegmentation(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection"""

    def __init__(self, config_file, checkpoint_file, image_dir=None, labels_file=None, score_threshold=0.3, device='cpu', **kwargs):
        """
        Load MMDetection model from config and checkpoint into memory.
        (Check https://mmdetection.readthedocs.io/en/v1.2.0/GETTING_STARTED.html#high-level-apis-for-testing-images)

        Optionally set mappings from COCO classes to target labels
        :param config_file: Absolute path to MMDetection config file (e.g. /home/user/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x.py)
        :param checkpoint_file: Absolute path MMDetection checkpoint file (e.g. /home/user/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth)
        :param image_dir: Directory where images are stored (should be used only in case you use direct file upload into Label Studio instead of URLs)
        :param labels_file: file with mappings from COCO labels to custom labels {"airplane": "Boeing"}
        :param score_threshold: score threshold to wipe out noisy results
        :param device: device (cpu, cuda:0, cuda:1, ...)
        :param kwargs:
        """
        super(MMSegmentation, self).__init__(**kwargs)

        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.labels_file = labels_file
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(f'{self.__class__.__name__} reads images from {self.image_dir}')
        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'PolygonLabels', 'Image')
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)

        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name

        print('Load new model from: ', config_file, checkpoint_file)
        self.model = init_segmentor(config_file, checkpoint_file, device=device)

    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as exc:
                logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        return image_url

    def predict(self, tasks, **kwargs):
        assert len(tasks) == 1
        task = tasks[0]
        image_url = self._get_image_url(task)
        image_path = get_image_local_path(image_url, image_dir=self.image_dir)
        model_results = inference_segmentor(self.model, image_path)
        results = []
        all_scores = []
        img_width, img_height = get_image_size(image_path)
        # TODO: model results is a segmentation map, need to convert to onehot?
        # import pdb; pdb.set_trace()
        for i, c in enumerate(self.model.CLASSES):
            points_list = Mask(model_results[0] == i).polygons().points
            output_label = self.label_map.get(c, c)

            if output_label not in self.labels_in_config:
                print(output_label + ' label not found in project config.')
                continue
            for points in points_list:
                points[:, 0] = points[:, 0] / img_width * 100
                points[:, 1] = points[:, 1] / img_height * 100
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'polygonlabels',
                    'value': {
                        'polygonlabels': [output_label],
                        'points': points.tolist()
                    },
                    'score': 0
                })
        return [{
            'result': results,
            'score': 0
        }]
