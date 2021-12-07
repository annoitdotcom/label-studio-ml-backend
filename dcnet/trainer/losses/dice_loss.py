import cv2
import torch
import torch.nn as nn

from dcnet.trainer.losses.loss_base import LossBase


class DiceLoss(LossBase):
    """ Loss function from https://arxiv.org/abs/1707.03237,
    where iou computation is introduced heatmap manner to measure the
    diversity bwtween tow heatmaps.
    """

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = float(eps)

    @classmethod
    def load_opt(cls, opt):
        return cls(
            eps=opt.optimize_settings.loss.eps
        )

    def _compute(self, pred, gt, mask, weights):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]

        assert pred.shape == gt.shape
        assert pred.shape == mask.shape

        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask

        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union

        assert loss <= 1
        return loss

    def forward(self, pred, batch):
        """
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of tow heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        """
        assert pred["thresh_binary"].dim() == 4, pred["thresh_binary"].dim()
        loss = self._compute(pred["thresh_binary"],
                             batch["gt"], batch["mask"], weights=None)
        return loss
