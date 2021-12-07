import torch
import torch.nn as nn

from dcnet.trainer.losses.dice_loss import DiceLoss
from dcnet.trainer.losses.loss_base import LossBase


class AdaptiveDiceLoss(LossBase):
    """ Integration of DiceLoss on 
    both binary prediction and thresh prediction.
    """

    def __init__(self, eps=1e-6):
        super(AdaptiveDiceLoss, self).__init__()
        self.main_loss = DiceLoss(eps)
        self.thresh_loss = DiceLoss(eps)

    @classmethod
    def load_opt(cls, opt):
        return cls(
            eps=opt.optimize_settings.loss.eps
        )

    def forward(self, pred, batch):
        assert isinstance(pred, dict)
        assert "binary" in pred
        assert "thresh_binary" in pred

        binary = pred["binary"]
        thresh_binary = pred["thresh_binary"]
        gt = batch["gt"]
        mask = batch["mask"]
        main_loss = self.main_loss(binary, gt, mask)
        thresh_loss = self.thresh_loss(thresh_binary, gt, mask)
        loss = main_loss + thresh_loss
        return (loss, dict(main_loss=main_loss, thresh_loss=thresh_loss))
