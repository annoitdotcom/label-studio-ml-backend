import torch
import torch.nn as nn

from dcnet.trainer.losses.balance_cross_entropy_loss import \
    BalanceCrossEntropyLoss
from dcnet.trainer.losses.dice_loss import DiceLoss
from dcnet.trainer.losses.loss_base import LossBase
from dcnet.trainer.losses.mask_l1_loss import MaskL1Loss


class L1BCEMiningLoss(LossBase):
    """ Basicly the same with L1BalanceCELoss, where the bce loss map is used as
        attention weigts for DiceLoss
    """

    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
        super(L1BCEMiningLoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    @classmethod
    def load_opt(cls, opt):
        return cls(
            eps=opt.optimize_settings.loss.eps,
            l1_scale=opt.optimize_settings.loss.l1_scale,
            bce_scale=opt.optimize_settings.loss.bce_scale
        )

    def forward(self, pred, batch):
        bce_loss, bce_map = self.bce_loss(pred["binary"], batch["gt"], batch["mask"],
                                          return_origin=True)
        l1_loss, l1_metric = self.l1_loss(
            pred["thresh"], batch["thresh_map"], batch["thresh_mask"])
        bce_map = (bce_map - bce_map.min()) / (bce_map.max() - bce_map.min())
        dice_loss = self.dice_loss(
            pred["thresh_binary"], batch["gt"],
            batch["mask"], weights=bce_map + 1)
        metrics = dict(bce_loss=bce_loss)
        metrics["thresh_loss"] = dice_loss
        loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
        metrics.update(**l1_metric)
        return (loss, metrics)
