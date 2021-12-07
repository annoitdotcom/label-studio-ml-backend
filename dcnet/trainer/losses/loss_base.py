import torch
import torch.nn as nn


class LossBase(nn.Module):
    """ Base loss class. Specifies the interface used by different loss types
    Input:
        x: A dict which contains predictions.
            thresh: The threshold prediction
            binary: The text segmentation prediction.
            thresh_binary: Value produced by `step_function(binary - thresh)`.
        y:
            gt: Text regions bitmap gt.
            mask: Ignore mask, pexels where value is 1 indicates no contribution to loss.
            thresh_mask: Mask indicates regions cared by thresh supervision.
            thresh_map: Threshold gt.
    Return:
        (loss, metrics).
        loss: A scalar loss value.
        metrics: A dict contraining partial loss values.
    """

    def __init__(self):
        super(LossBase, self).__init__()

    @classmethod
    def load_opt(cls, opt):
        raise NotImplementedError

    def forward(self, x, y):
        raise NotImplementedError
