import torch
import torch.nn as nn

from dcnet.trainer.losses.adaptive_dice_loss import AdaptiveDiceLoss
from dcnet.trainer.losses.loss_base import LossBase
from dcnet.trainer.losses.mask_l1_loss import MaskL1Loss


class L1DiceLoss(LossBase):
    """ L1Loss on thresh, DiceLoss on thresh_binary and binary.
    """

    def __init__(self, eps=1e-6, l1_scale=10):
        super(L1DiceLoss, self).__init__()
        self.dice_loss = AdaptiveDiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.l1_scale = l1_scale

    @classmethod
    def load_opt(cls, opt):
        return cls(
            eps=opt.optimize_settings.loss.eps,
            l1_scale=opt.optimize_settings.loss.l1_scale
        )

    def forward(self, pred, batch):
        dice_loss, metrics = self.dice_loss(pred, batch)
        l1_loss, l1_metric = self.l1_loss(
            pred["thresh"], batch["thresh_map"], batch["thresh_mask"])

        loss = dice_loss + self.l1_scale * l1_loss
        metrics.update(**l1_metric)
        return (loss, metrics)
