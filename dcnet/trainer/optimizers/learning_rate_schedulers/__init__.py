"""Module defining learning rate scheduler."""
from ....trainer.optimizers.learning_rate_schedulers.built_in_learning_rate import \
    BuitlinLearningRate
from ....trainer.optimizers.learning_rate_schedulers.constant_learning_rate import \
    ConstantLearningRate
from ....trainer.optimizers.learning_rate_schedulers.decay_learning_rate import \
    DecayLearningRate
from ....trainer.optimizers.learning_rate_schedulers.learning_rate_base import \
    LearningRateSchedulerBase
from ....trainer.optimizers.learning_rate_schedulers.multi_step_learning_rate import \
    MultiStepLearningRate
from ....trainer.optimizers.learning_rate_schedulers.piece_wise_constant_learning_rate import \
    PiecewiseConstantLearningRate
from ....trainer.optimizers.learning_rate_schedulers.warmup_learning_rate import \
    WarmupLearningRate

str2lr = {
    "builtin_learning_rate": BuitlinLearningRate,
    "constant_learning_rate": ConstantLearningRate,
    "decay_learning_rate": DecayLearningRate,
    "multi_step_learning_rate": MultiStepLearningRate,
    "piece_wiseLconstant_learning_rate": PiecewiseConstantLearningRate,
    "warmup_learning_rate": WarmupLearningRate,

}

__all__ = [
    "LearningRateSchedulerBase",
    "BuitlinLearningRate",
    "ConstantLearningRate",
    "DecayLearningRate",
    "MultiStepLearningRate",
    "PiecewiseConstantLearningRate",
    "WarmupLearningRate",
    "str2lr"
]
