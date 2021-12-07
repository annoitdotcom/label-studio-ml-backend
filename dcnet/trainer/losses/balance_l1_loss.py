import torch
import torch.nn as nn

from dcnet.trainer.losses.loss_base import LossBase


class BalanceL1Loss(LossBase):
    def __init__(self, negative_ratio=3.):
        super(BalanceL1Loss, self).__init__()
        self.negative_ratio = negative_ratio

    @classmethod
    def load_opt(cls, opt):
        return cls(
            opt.optimize_settings.loss.negative_ratio
        )

    def _compute(self, pred: torch.Tensor, gt, mask):
        """
        Args:
            pred: (N, 1, H, W).
            gt: (N, H, W).
            mask: (N, H, W).
        """
        loss = torch.abs(pred[:, 0] - gt)
        positive = loss * mask
        negative = loss * (1 - mask)
        positive_count = int(mask.sum())
        negative_count = min(
            int((1 - mask).sum()),
            int(positive_count * self.negative_ratio))
        negative_loss, _ = torch.topk(negative.view(-1), negative_count)
        negative_loss = negative_loss.sum() / negative_count
        positive_loss = positive.sum() / positive_count
        return positive_loss + negative_loss,\
            dict(l1_loss=positive_loss, nge_l1_loss=negative_loss)

    def forward(self, pred, batch):
        loss = self._compute(pred, batch["gt"], batch["mask"])
        return loss
