import cv2
import torch
import torch.nn as nn

from dcnet.trainer.losses.loss_base import LossBase


class LeakyDiceLoss(LossBase):
    """
    Variation from DiceLoss.
    The coverage and union are computed separately.
    """

    def __init__(self, eps=1e-6, coverage_scale=5.0):
        super(LeakyDiceLoss, self).__init__()
        self.eps = eps
        self.coverage_scale = coverage_scale

    @classmethod
    def load_opt(cls, opt):
        return cls(
            eps=opt.optimize_settings.loss.eps,
            coverage_scale=opt.optimize_settings.loss.coverage_scale
        )

    def forward(self, pred, batch):
        gt = batch["gt"]
        mask = batch["mask"]

        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]

        assert pred.shape == gt.shape
        assert pred.shape == mask.shape

        coverage = (pred * mask * gt).sum() / ((gt * mask).sum() + self.eps)
        assert coverage <= 1

        coverage = 1 - coverage
        excede = (pred * mask * gt).sum() / ((pred * mask).sum() + self.eps)
        assert excede <= 1

        excede = 1 - excede
        loss = coverage * self.coverage_scale + excede
        return (loss, dict(coverage=coverage, excede=excede))
