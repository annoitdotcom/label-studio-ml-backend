import torch.nn as nn


class DatasetBase(nn.Module):
    """Processes of data dict. """

    def __init__(self):
        super(DatasetBase, self).__init__()

    @classmethod
    def load_opt(cls, opt):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
