import numpy as np
import torch

from dcnet.dataloader.preprocess.preprocess_base import PreprocessBase


class NormalizeImage(PreprocessBase):
    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])

    def __init__(self, opt):
        super(NormalizeImage, self).__init__()
        self.opt = opt

    def __call__(self, data):
        assert "image" in data, "`image` in data is required by this process"
        image = data["image"]
        image -= self.RGB_MEAN
        image /= 255.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        data["image"] = image
        return data

    @classmethod
    def restore(self, image):
        image = image.permute(1, 2, 0).to("cpu").numpy()
        image = image * 255.
        image += self.RGB_MEAN
        image = image.astype(np.uint8)
        return image

    @classmethod
    def load_opt(cls, opt, is_training):
        return cls(opt)
