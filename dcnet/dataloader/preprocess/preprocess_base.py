import torch.nn as nn


class PreprocessBase(nn.Module):
    """Processes of data dict."""

    def __init__(self):
        super(PreprocessBase, self).__init__()

    @classmethod
    def load_opt(cls, opt, is_training):
        raise NotImplementedError

    def __call__(self, data):
        raise NotImplementedError

    def process(self, data):
        raise NotImplementedError
