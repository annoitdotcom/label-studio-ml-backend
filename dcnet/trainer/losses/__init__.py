"""Module defining losses."""
from dcnet.trainer.losses.adaptive_dice_loss import AdaptiveDiceLoss
from dcnet.trainer.losses.adaptive_instance_dice_loss import \
    AdaptiveInstanceDiceLoss
from dcnet.trainer.losses.balance_cross_entropy_loss import \
    BalanceCrossEntropyLoss
from dcnet.trainer.losses.balance_l1_loss import BalanceL1Loss
from dcnet.trainer.losses.dice_loss import DiceLoss
from dcnet.trainer.losses.l1_balance_ce_loss import L1BalanceCELoss
from dcnet.trainer.losses.l1_bce_mining_loss import L1BCEMiningLoss
from dcnet.trainer.losses.l1_dice_loss import L1DiceLoss
from dcnet.trainer.losses.l1_leaky_dice_loss import L1LeakyDiceLoss
from dcnet.trainer.losses.leaky_dice_loss import LeakyDiceLoss
from dcnet.trainer.losses.loss_base import LossBase
from dcnet.trainer.losses.mask_l1_loss import MaskL1Loss

str2loss = {
    "dice_loss": DiceLoss,
    "balance_cross_entropy_loss": BalanceCrossEntropyLoss,
    "adaptive_dice_loss": AdaptiveDiceLoss,
    "balance_l1_loss": BalanceL1Loss,
    "adaptive_instance_dice_loss": AdaptiveInstanceDiceLoss,
    "l1_dice_loss": L1DiceLoss,
    "l1_balance_ce_loss": L1BalanceCELoss,
    "l1_bce_mining_loss": L1BCEMiningLoss,
    "l1_leaky_dice_loss": L1LeakyDiceLoss,
    "leaky_dice_loss": LeakyDiceLoss,
    "mask_l1_loss": MaskL1Loss,
}


__all__ = [
    "LossBase",
    "str2loss",
    "DiceLoss",
    "BalanceCrossEntropyLoss",
    "AdaptiveDiceLoss",
    "BalanceL1Loss",
    "AdaptiveInstanceDiceLoss",
    "L1DiceLoss",
    "L1BalanceCELoss",
    "L1BCEMiningLoss",
    "L1LeakyDiceLoss",
    "LeakyDiceLoss",
    "MaskL1Loss",

]
