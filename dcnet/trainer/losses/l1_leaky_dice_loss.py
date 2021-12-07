import torch
import torch.nn as nn

from dcnet.trainer.losses.dice_loss import DiceLoss
from dcnet.trainer.losses.leaky_dice_loss import LeakyDiceLoss
from dcnet.trainer.losses.loss_base import LossBase
from dcnet.trainer.losses.mask_l1_loss import MaskL1Loss


class L1LeakyDiceLoss(LossBase):
    """ LeakyDiceLoss on binary,
    MaskL1Loss on thresh,
    DiceLoss on thresh_binary.
    """

    def __init__(self, eps=1e-6, coverage_scale=5, l1_scale=10):
        super(L1LeakyDiceLoss, self).__init__()
        self.l1_loss = MaskL1Loss()
        self.thresh_loss = DiceLoss(eps=eps)
        self.main_loss = LeakyDiceLoss(coverage_scale=coverage_scale)
        self.l1_scale = l1_scale

    @classmethod
    def load_opt(cls, opt):
        return cls(
            eps=opt.optimize_settings.loss.eps,
            coverage_scale=opt.optimize_settings.loss.coverage_scale,
            l1_scale=opt.optimize_settings.loss.l1_scale
        )

    def forward(self, pred, batch):
        main_loss, metrics = self.main_loss(
            pred["binary"], batch["gt"], batch["mask"])
        thresh_loss = self.thresh_loss(
            pred["thresh_binary"], batch["gt"], batch["mask"])
        l1_loss, l1_metric = self.l1_loss(
            pred["thresh"], batch["thresh_map"], batch["thresh_mask"])
        metrics.update(**l1_metric, thresh_loss=thresh_loss)
        loss = main_loss + thresh_loss + l1_loss * self.l1_scale
        return (loss, metrics)
