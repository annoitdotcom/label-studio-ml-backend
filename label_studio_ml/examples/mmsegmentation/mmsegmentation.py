import hashlib
import io
import logging
import os
import urllib
from urllib.parse import urlparse

import boto3
import mmcv
import numpy as np
import requests
from botocore.exceptions import ClientError
from imantics import Mask, Polygons
from label_studio.core.settings.base import DATA_UNDEFINED_NAME
from label_studio.core.utils.io import get_data_dir, json_load
from mmcv import Config
from mmseg.apis import (inference_segmentor, init_segmentor, set_random_seed,
                        train_segmentor)
from mmseg.datasets import build_dataset
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.models import build_segmentor
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (get_choice, get_image_local_path,
                                   get_image_size, get_single_tag_keys,
                                   is_skipped)

logger = logging.getLogger(__name__)
image_size = 224
image_cache_dir = os.path.join(os.path.dirname(__file__), 'image-cache')
os.makedirs(image_cache_dir, exist_ok=True)


class MMSegmentation(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection"""

    def __init__(self, config_file, checkpoint_file, image_dir=None, labels_file=None,
                 ann_dir=None, score_threshold=0.3, device='cpu', **kwargs):
        """
        Load MMDetection model from config and checkpoint into memory.
        (Check https://mmdetection.readthedocs.io/en/v1.2.0/GETTING_STARTED.html#high-level-apis-for-testing-images)

        Optionally set mappings from COCO classes to target labels
        :param config_file: Absolute path to MMDetection config file (e.g. /home/user/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x.py)
        :param checkpoint_file: Absolute path MMDetection checkpoint file (e.g. /home/user/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth)
        :param image_dir: Directory where images are stored (should be used only in case you use direct file upload into Label Studio instead of URLs)
        :param ann_dir: Directory where segmentation masks are stored (should be used only in case you use direct file upload into Label Studio instead of URLs)
        :param labels_file: file with mappings from COCO labels to custom labels {"airplane": "Boeing"}
        :param score_threshold: score threshold to wipe out noisy results
        :param device: device (cpu, cuda:0, cuda:1, ...)
        :param kwargs:
        """
        super(MMSegmentation, self).__init__(**kwargs)

        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.labels_file = labels_file
        # Default Label Studio image upload folder.
        self.ann_dir = ann_dir
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

        logger.debug('Load new model from: ', self.config_file, self.checkpoint_file)
        if not self.train_output:
            # This is an array of <Choice> labels
            self.labels = self.info['labels']

            # If there is no trainings, load default model.
            logger.debug(f'Initialized with config file={self.config_file}, checkpoint_file={self.checkpoint_file}')
        else:
            # Otherwise load the model from the latest training results.
            self.checkpoint_file = self.train_output['checkpoint_file']
            self.config_file = self.train_output['config_file']
        
            # and use the labels from training outputs
            self.labels = self.train_output['labels']
            logger.debug(f'Loaded from train output with config file={self.config_file}, checkpoint file={checkpoint_file}')
        self.model = init_segmentor(self.config_file, checkpoint_file, device=device)
        self.cfg = self.get_training_cfg(self.config_file)

    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # presign s3 url
            rr = urlparse(image_url, allow_fragments=False)
            bucket_name = rr.netloc
            key = rr.path.lstrip('/')
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
        for idx, cc in enumerate(self.model.CLASSES):
            points_list = Mask(model_results[0] == (idx + 1)).polygons().points
            output_label = self.label_map.get(cc, cc)
            if output_label not in self.labels_in_config:
                logger.error(output_label + ' label not found in project config.')
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

    @DATASETS.register_module()
    class SegmentationDataset(CustomDataset):
        def __init__(self, split, **kwargs):
            super().__init__(img_suffix='.jpg', seg_map_suffix='.png', 
                            split=split, **kwargs)
            assert os.path.exists(self.img_dir) and self.split is not None
    
    def get_training_cfg(self, config_file: str = None, image_dir: str = None, ann_dir: str = None):
        cfg = Config.fromfile(config_file)

        # Since we use ony one GPU, BN is used instead of SyncBN
        cfg.norm_cfg = dict(type='BN', requires_grad=True)
        cfg.model.backbone.norm_cfg = cfg.norm_cfg
        cfg.model.decode_head.norm_cfg = cfg.norm_cfg
        cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
        # modify num classes of the model in decode/auxiliary head
        cfg.model.decode_head.num_classes=8
        cfg.model.auxiliary_head.num_classes=8

        # Modify dataset type and path
        cfg.dataset_type = 'SegmentationDataset'
        cfg.data.samples_per_gpu = 4
        cfg.data.workers_per_gpu = 4
        cfg.img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        cfg.crop_size = (256, 256)
        cfg.train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(320, 240), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        ]

        cfg.data.train.type = cfg.dataset_type
        cfg.data.train.img_dir = image_dir
        cfg.data.train.ann_dir = ann_dir
        cfg.data.train.pipeline = cfg.train_pipeline

        # We can still use the pre-trained Mask RCNN model though we do not need to
        # use the mask branch
        cfg.load_from = self.checkpoint_file

        # Set up working dir to save files and logs.
        cfg.work_dir = None

        cfg.runner.max_iters = 1000
        cfg.log_config.interval = 10
        cfg.evaluation.interval = 200
        cfg.checkpoint_config.interval = 200

        # Set seed to facitate reproducing the result.
        cfg.seed = 0
        set_random_seed(0, deterministic=False)
        cfg.gpu_ids = range(1)

        # Let's have a look at the final config used for training.
        logger.debug(f'Config:\n{cfg.pretty_text}')
        return cfg

    def fit(self, completions, workdir=None, **kwargs):
        annotations = []
        for completion in completions:
            if is_skipped(completion):
                continue

            image_url = self._get_image_url(completion['data'][self.value])
            image_path = get_image_local_path(image_url, image_dir=self.image_dir)
            ann_url = get_choice(completion)
            ann_path = get_image_local_path(ann_url, image_dir=self.ann_dir)
            annotations.append((image_path, ann_path))

        # Create datasets.
        image_paths, ann_paths = zip(*annotations)
        palette = (tuple(np.random.choice(range(256), size=3)) for _ in range(len(self.labels)))
        CustomDataset.CLASSES = self.labels
        CustomDataset.PALETTE = palette

        # Get training config.
        cfg = self.get_training_cfg(self.config_file, self.image_dir, self.ann_dir)
        cfg.workdir = workdir

        # Build the datasets.
        datasets = [build_dataset(cfg.data.train)]

        # Build the detector.
        model = build_segmentor(self.model, train_cfg=self.cfg.get('train_cfg'))

        # Add an attribute for visualization convenience.
        model.CLASSES = datasets[0].CLASSES

        # Create work_dir.
        mmcv.mkdir_or_exist(os.path.abspath(self.cfg.work_dir))
        train_segmentor(model, datasets, self.cfg, distributed=False, validate=False, meta=dict())
        train_output = {
            'checkpoint_file': os.path.join(self.cfg.work_dir, 'latest.pth'),
            'config_file': os.path.join(self.cfg.work_dir, os.path.basename(self.config_file))
        }
        return train_output
