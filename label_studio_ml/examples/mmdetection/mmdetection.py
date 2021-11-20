import logging
import os
import os.path as osp
from urllib.parse import urlparse

import boto3
import mmcv
import numpy as np
from botocore.exceptions import ClientError
from label_studio.core.settings.base import DATA_UNDEFINED_NAME
from label_studio.core.utils.io import get_data_dir, json_load
from mmcv import Config
from mmdet.apis import (inference_detector, init_detector, set_random_seed,
                        train_detector)
from mmdet.datasets import build_dataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.models import build_detector

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (get_choice, get_image_local_path,
                                   get_image_size, get_single_tag_keys,
                                   is_skipped)

logger = logging.getLogger(__name__)


@DATASETS.register_module()
class DetectionDataset(CustomDataset):
    def load_annotations(self, ann_file, label_dir: str = None, img_suffix=".jpg"):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # Load image list from file.
        image_list = mmcv.list_from_file(ann_file)
        data_infos = []
        # Convert annotations to middle format.
        for image_id in image_list:
            filename = f'{self.img_prefix}/{image_id}{img_suffix}'
            image = mmcv.imread(filename)
            height, width = image.shape[:2]
            data_info = dict(
                filename=f'{image_id}{img_suffix}', width=width, height=height)

            # Load annotations
            lines = mmcv.list_from_file(osp.join(label_dir, f'{image_id}.txt'))
            content = [line.strip().split(' ') for line in lines]
            bbox_names = [x[0] for x in content]
            bboxes = [[float(info) for info in x[4:8]] for x in content]

            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []

            # filter 'DontCare'
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
                    gt_bboxes.append(bbox)
                else:
                    gt_labels_ignore.append(-1)
                    gt_bboxes_ignore.append(bbox)

            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.long),
                bboxes_ignore=np.array(gt_bboxes_ignore,
                                       dtype=np.float32).reshape(-1, 4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.long))

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos


class MMDetection(LabelStudioMLBase):
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
        super(MMDetection, self).__init__(**kwargs)

        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.labels_file = labels_file
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(
            f'{self.__class__.__name__} reads images from {self.image_dir}')
        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)

        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name

        print('Load new model from: ', config_file, checkpoint_file)
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.score_thresh = score_threshold

    def _get_image_url(self, task):
        image_url = task['data'].get(
            self.value) or task['data'].get(DATA_UNDEFINED_NAME)
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
                logger.warning(
                    f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        return image_url

    def predict(self, tasks, **kwargs):
        assert len(tasks) == 1
        task = tasks[0]
        image_url = self._get_image_url(task)
        image_path = get_image_local_path(image_url, image_dir=self.image_dir)
        model_results = inference_detector(self.model, image_path)
        results = []
        all_scores = []
        img_width, img_height = get_image_size(image_path)
        for bboxes, label in zip(model_results, self.model.CLASSES):
            output_label = self.label_map.get(label, label)

            if output_label not in self.labels_in_config:
                print(output_label + ' label not found in project config.')
                continue
            for bbox in bboxes:
                bbox = list(bbox)
                if not bbox:
                    continue
                score = float(bbox[-1])
                if score < self.score_thresh:
                    continue
                x, y, xmax, ymax = bbox[:4]
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [output_label],
                        'x': x / img_width * 100,
                        'y': y / img_height * 100,
                        'width': (xmax - x) / img_width * 100,
                        'height': (ymax - y) / img_height * 100
                    },
                    'score': score
                })
                all_scores.append(score)
        avg_score = sum(all_scores) / max(len(all_scores), 1)
        return [{
            'result': results,
            'score': avg_score
        }]

    def get_training_cfg(self, ann_file, num_classes):
        cfg = Config.fromfile(self.config_file)

        # Modify dataset type and path
        cfg.dataset_type = 'DetectionDataset'
        cfg.data.train.type = 'DetectionDataset'
        cfg.data.train.ann_file = ann_file
        cfg.data.train.img_prefix = self.image_dir

        # modify num classes of the model in box head
        cfg.model.roi_head.bbox_head.num_classes = num_classes
        # We can still use the pre-trained Mask RCNN model though we do not need to
        # use the mask branch
        cfg.load_from = self.checkpoint_file

        # Set up working dir to save files and logs.
        cfg.work_dir = None

        # The original learning rate (LR) is set for 8-GPU training.
        # We divide it by 8 since we only use one GPU.
        cfg.optimizer.lr = 0.02 / 8
        cfg.lr_config.warmup = None
        cfg.log_config.interval = 10

        # Change the evaluation metric since we use customized dataset.
        cfg.evaluation.metric = 'mAP'
        # We can set the evaluation interval to reduce the evaluation times
        cfg.evaluation.interval = 12
        # We can set the checkpoint saving interval to reduce the storage cost
        cfg.checkpoint_config.interval = 12

        # Set seed thus the results are more reproducible
        cfg.seed = 0
        set_random_seed(0, deterministic=False)
        cfg.gpu_ids = range(1)

        # We can initialize the logger for training and have a look
        # at the final config used for training
        print(f'Config:\n{cfg.pretty_text}')

    def fit(self, completions, workdir=None, **kwargs):
        annotations = []
        for completion in completions:
            if is_skipped(completion):
                continue

            image_url = self._get_image_url(completion['data'][self.value])
            image_path = get_image_local_path(
                image_url, image_dir=self.image_dir)
            ann_url = get_choice(completion)
            ann_path = get_image_local_path(ann_url, image_dir=self.ann_dir)
            annotations.append((image_path, ann_path))

        # Create datasets.
        image_paths, ann_paths = zip(*annotations)
        CustomDataset.CLASSES = self.labels

        # Get training config.
        # TODO: set ann file here.
        cfg = self.get_training_cfg(self.labels_file, len(self.labels))
        cfg.workdir = workdir

        # Build the datasets.
        datasets = [build_dataset(cfg.data.train)]

        # Build the detector.
        model = build_detector(self.model, train_cfg=self.cfg.get('train_cfg'))

        # Add an attribute for visualization convenience.
        model.CLASSES = datasets[0].CLASSES

        # Create work_dir.
        mmcv.mkdir_or_exist(os.path.abspath(self.cfg.work_dir))
        train_detector(model, datasets, self.cfg,
                       distributed=False, validate=False, meta=dict())
        train_output = {
            'checkpoint_file': os.path.join(self.cfg.work_dir, 'latest.pth'),
            'config_file': os.path.join(self.cfg.work_dir, os.path.basename(self.config_file))
        }
        return train_output
