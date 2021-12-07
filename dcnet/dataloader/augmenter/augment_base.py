import torch.nn as nn


class AugmenterBase(nn.Module):
    """Processes of data dict. """

    @classmethod
    def load_opt(cls, opt):
        raise NotImplementedError

    def process(self, data):
        raise NotImplementedError

    def __call__(self, data):
        raise NotImplementedError
