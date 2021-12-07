import torch
import torch.nn as nn

from dcnet.trainer.losses.adaptive_dice_loss import AdaptiveDiceLoss
from dcnet.trainer.losses.balance_l1_loss import BalanceL1Loss
from dcnet.trainer.losses.loss_base import LossBase


class FullL1DiceLoss(L1DiceLoss):
    """
    L1loss on thresh, pixels with topk losses in non-text regions are also counted.
    DiceLoss on thresh_binary and binary.
    """

    def __init__(self, eps=1e-6, l1_scale=10):
        nn.Module.__init__(self)
        self.dice_loss = AdaptiveDiceLoss(eps=eps)
        self.l1_loss = BalanceL1Loss()
        self.l1_scale = l1_scale

    @classmethod
    def load_opt(cls, opt):
        return cls(
            eps=opt.optimize_settings.loss.eps,
            l1_scale=opt.optimize_settings.loss.l1_scale
        )
