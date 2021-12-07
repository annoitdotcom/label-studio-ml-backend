"""Module defining encoders."""
from dcnet.dataloader.augmenter.augment_base import AugmenterBase
from dcnet.dataloader.augmenter.augment_data import AugmentDetectionData
from dcnet.dataloader.augmenter.random_crop_data import RandomCropData

str2augm = {
    "augment_data": AugmentDetectionData,
    "random_crop_data": RandomCropData
}

__all__ = ["AugmentDetectionData",
           "RandomCropData", "AugmenterBase", "str2augm"]
