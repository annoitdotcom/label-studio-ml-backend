"""Module defining encoders."""
from dcnet.dataloader.preprocess.filter_keys import FilterKeys
from dcnet.dataloader.preprocess.make_border_map import MakeBorderMap
from dcnet.dataloader.preprocess.make_icdar_data import MakeICDARData
from dcnet.dataloader.preprocess.make_seg_detection_data import \
    MakeSegDetectionData
from dcnet.dataloader.preprocess.make_validation_data import MakeValidationData
from dcnet.dataloader.preprocess.normalize_image import NormalizeImage

str2preprocess = {
    "make_icdar_data": MakeICDARData,
    "make_seg_detection_data": MakeSegDetectionData,
    "make_border_map": MakeBorderMap,
    "filter_keys": FilterKeys,
    "normalize_image": NormalizeImage,
    "make_validation_data": MakeValidationData
}

__all__ = ["MakeValidationData", "MakeSegDetectionData", "MakeICDARData",
           "MakeBorderMap", "FilterKeys", "NormalizeImage", "str2preprocess"]
