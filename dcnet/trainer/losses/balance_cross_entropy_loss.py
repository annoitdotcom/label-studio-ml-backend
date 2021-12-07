import cv2
import torch
import torch.nn as nn

from dcnet.trainer.losses.loss_base import LossBase


class BalanceCrossEntropyLoss(LossBase):
    """ Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """

    def __init__(self, eps=1e-6, negative_ratio=3.0):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.eps = eps
        self.negative_ratio = negative_ratio

    @classmethod
    def load_opt(cls, opt):
        return cls(
            eps=opt.optimize_settings.loss.eps
        )

    def get_loss(self,
                 pred: torch.Tensor,
                 gt: torch.Tensor,
                 mask: torch.Tensor,
                 return_origin=False):
        """
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        """
        # cv2.imwrite("mask_pred.jpg", pred.cpu().detach().numpy()[0][0] * 255)
        # cv2.imwrite("mask_groundtruth.jpg", gt.cpu().detach().numpy()[0][0] * 255)

        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()),
                             int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(
            pred, gt, reduction="none")[:, 0, :, :]
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) /\
            (positive_count + negative_count + self.eps)

        if return_origin:
            return balance_loss, loss
        return balance_loss

    def forward(self, pred, batch):
        loss = self.get_loss(pred["binary"], batch["gt"], batch["mask"])
        return loss
