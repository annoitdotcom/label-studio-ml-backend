"""Module defining encoders."""
from dcnet.dataloader.dataset.datapile_dataset import DatapileDataset
from dcnet.dataloader.dataset.dataset_base import DatasetBase
from dcnet.dataloader.dataset.image_dataset import ImageDataset

str2dataset = {
    "image_dataset": ImageDataset,
    "datapile_dataset": DatapileDataset
}

__all__ = ["ImageDataset", "DatasetBase", "DatapileDataset", "str2dataset"]
