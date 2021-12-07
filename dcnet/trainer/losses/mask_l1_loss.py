import torch
import torch.nn as nn

from dcnet.trainer.losses.loss_base import LossBase


class MaskL1Loss(LossBase):
    def __init__(self):
        super(MaskL1Loss, self).__init__()

    @classmethod
    def load_opt(cls, opt):
        raise NotImplementedError

    def forward(self, pred: torch.Tensor, gt, mask):
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            return (mask_sum, dict(l1_loss=mask_sum))
        else:
            loss = (torch.abs(pred[:, 0] - gt) * mask).sum() / mask_sum
            return (loss, dict(l1_loss=loss))
