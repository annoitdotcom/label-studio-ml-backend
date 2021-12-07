import torch
import torch.nn as nn

from dcnet.trainer.losses.dice_loss import DiceLoss
from dcnet.trainer.losses.instance_dice_loss import InstanceDiceLoss
from dcnet.trainer.losses.loss_base import LossBase


class AdaptiveInstanceDiceLoss(LossBase):
    """ InstanceDiceLoss on both binary and thresh_bianry.
    """

    def __init__(self, iou_thresh=0.2, thresh=0.3):
        super(AdaptiveInstanceDiceLoss, self).__init__()
        self.main_loss = DiceLoss()
        self.main_instance_loss = InstanceDiceLoss()

        self.thresh_loss = DiceLoss()
        self.thresh_instance_loss = InstanceDiceLoss()

        self.weights = nn.ParameterDict(dict(
            main=nn.Parameter(torch.ones(1)),
            thresh=nn.Parameter(torch.ones(1)),
            main_instance=nn.Parameter(torch.ones(1)),
            thresh_instance=nn.Parameter(torch.ones(1))))

    @classmethod
    def load_opt(cls, opt):
        return cls(
            iou_thresh=opt.optimize_settings.loss.iou_thresh,
            thresh=opt.optimize_settings.loss.thresh
        )

    def partial_loss(self, weight, loss):
        return loss / weight + torch.log(torch.sqrt(weight))

    def forward(self, pred, batch):
        main_loss = self.main_loss(pred["binary"], batch["gt"], batch["mask"])
        thresh_loss = self.thresh_loss(
            pred["thresh_binary"], batch["gt"], batch["mask"])
        main_instance_loss = self.main_instance_loss(
            pred["binary"], batch["gt"], batch["mask"])
        thresh_instance_loss = self.thresh_instance_loss(
            pred["thresh_binary"], batch["gt"], batch["mask"])
        loss = self.partial_loss(self.weights["main"], main_loss) \
            + self.partial_loss(self.weights["thresh"], thresh_loss) \
            + self.partial_loss(self.weights["main_instance"], main_instance_loss) \
            + self.partial_loss(self.weights["thresh_instance"], thresh_instance_loss)
        metrics = dict(
            main_loss=main_loss,
            thresh_loss=thresh_loss,
            main_instance_loss=main_instance_loss,
            thresh_instance_loss=thresh_instance_loss)
        metrics.update(self.weights)
        return (loss, metrics)
