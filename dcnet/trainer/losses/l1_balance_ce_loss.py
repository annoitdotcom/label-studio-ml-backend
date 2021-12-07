import torch
import torch.nn as nn

from dcnet.trainer.losses.balance_cross_entropy_loss import \
    BalanceCrossEntropyLoss
from dcnet.trainer.losses.dice_loss import DiceLoss
from dcnet.trainer.losses.loss_base import LossBase
from dcnet.trainer.losses.mask_l1_loss import MaskL1Loss


class L1BalanceCELoss(LossBase):
    """ Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    """

    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
        super(L1BalanceCELoss, self).__init__()
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
        bce_loss = self.bce_loss(pred, batch)
        metrics = dict(bce_loss=bce_loss)

        if "thresh" in pred:
            l1_loss, l1_metric = self.l1_loss(
                pred["thresh"], batch["thresh_map"], batch["thresh_mask"])
            dice_loss = self.dice_loss(pred, batch)
            metrics["thresh_loss"] = dice_loss
            loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
            metrics.update(**l1_metric)
        else:
            loss = bce_loss
        return (loss, metrics)
