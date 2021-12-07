import numpy as np

from dcnet.trainer.optimizers.learning_rate_schedulers.learning_rate_base import \
    LearningRateSchedulerBase


class DecayLearningRate(LearningRateSchedulerBase):
    """

    """

    def __init__(self, learning_rate=0.002, factor=0.9, epochs=200):
        self.learning_rate = learning_rate
        self.factor = factor
        self.epochs = epochs

    @classmethod
    def load_opt(cls, opt):
        return cls(
            opt.optimize_settings.learning_rate_scheduler.learning_rate,
            opt.optimize_settings.learning_rate_scheduler.factor,
            opt.optimize_settings.epochs
        )

    def get_learning_rate(self, epoch, step=None):
        rate = np.power(1.0 - epoch / float(self.epochs + 1), self.factor)
        return rate * self.learning_rate
