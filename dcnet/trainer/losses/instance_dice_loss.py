import cv2
import torch
import torch.nn as nn
from scipy import ndimage

from dcnet.trainer.losses.dice_loss import DiceLoss


class InstanceDiceLoss(DiceLoss):
    """
    DiceLoss normalized on each instance.
    Input:
        pred: (N, 1, H, W)
        gt: (N, 1, H, W)
        mask: (N, H, W)
    Note: This class assume that input tensors are on gpu,
        while cput computation is required to find union areas.
    """
    REDUCTION = ["mean", "sum", "none"]

    def __init__(self, threshold=0.3, iou_thresh=0.2, reduction=None,
                 max_regions=100, eps=1e-6):
        nn.Module.__init__(self)
        self.threshold = threshold
        self.iou_thresh = iou_thresh
        self.reduction = reduction
        if self.reduction is None:
            self.reduction = "mean"
        assert self.reduction in self.REDUCTION
        self.max_regions = max_regions
        self.eps = eps

    def label(self, tensor_on_gpu, blur=None):
        """
        Args:
            tensor_on_gpu: (N, 1, H, W)
            blur: Lambda. If exists, each instance will be blured using `blur`.
        """
        tensor = tensor_on_gpu.cpu().detach().numpy()

        instance_maps = []
        instance_counts = []
        for batch_index in range(tensor_on_gpu.shape[0]):
            instance = tensor[batch_index]
            if blur is not None:
                instance = blur(instance)
            lable_map, instance_count = ndimage.label(instance[0])
            instance_count = min(self.max_regions, instance_count)
            instance_map = []
            for index in range(1, instance_count):
                instance = torch.from_numpy(
                    lable_map == index).to(tensor_on_gpu.device).type(torch.float32)
                instance_map.append(instance)
            instance_maps.append(instance_map)
        return instance_maps, instance_counts

    def iou(self, pred, gt):
        overlap = (pred * gt).sum()
        return max(overlap / pred.sum(), overlap / gt.sum())

    def replace_or_add(self, dest, value):
        if dest is None:
            return value
        if value is None:
            return dest
        return dest + value

    def forward(self, pred, gt, mask):
        # pred_label_maps: N, P, H, W, where P is the number of regions.
        torch.cuda.synchronize()
        pred_label_maps, _ = self.label(pred > self.threshold)
        gt_label_maps, _ = self.label(gt)

        losses = []
        for batch_index, gt_instance_maps in enumerate(gt_label_maps):
            pred_instance_maps = pred_label_maps[batch_index]
            if gt_instance_maps is None or pred_instance_maps is None:
                continue

            single_loss = None  # loss on a single image in a batch
            mask_not_matched = set(range(len(pred_instance_maps)))
            for gt_instance_map in gt_instance_maps:
                instance_loss = None  # loss on a specific gt region
                for instance_index, pred_instance_map in enumerate(pred_instance_maps):
                    if self.iou(pred_instance_map, gt_instance_map) > self.iou_thresh:
                        match_loss = self._compute(
                            pred[batch_index][0], gt[batch_index][0],
                            mask[batch_index] * (pred_instance_map + gt_instance_map > 0).type(torch.float32))
                        instance_loss = self.replace_or_add(
                            instance_loss, match_loss)
                        if instance_index in mask_not_matched:
                            mask_not_matched.remove(instance_index)
                if instance_loss is None:
                    instance_loss = self._compute(
                        pred[batch_index][0], gt[batch_index][0],
                        mask[batch_index] * gt_instance_map)
                single_loss = self.replace_or_add(single_loss, instance_loss)

            """Whether to compute single loss on instances which contrain no positive sample.
            if single_loss is None:
                single_loss = self._compute(
                        pred[batch_index][0], gt[batch_index][0],
                        mask[batch_index])
            """

            for instance_index in mask_not_matched:
                single_loss = self.replace_or_add(
                    single_loss,
                    self._compute(
                        pred[batch_index][0], gt[batch_index][0],
                        mask[batch_index] * pred_instance_maps[instance_index]))

            if single_loss is not None:
                losses.append(single_loss)

        if self.reduction == "none":
            loss = losses
        else:
            assert self.reduction in ["sum", "mean"]
            count = len(losses)
            loss = sum(losses)
            if self.reduction == "mean":
                loss = loss / count
        return loss
