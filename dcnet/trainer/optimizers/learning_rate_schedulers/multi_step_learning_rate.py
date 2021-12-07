from bisect import bisect_right

from dcnet.trainer.optimizers.learning_rate_schedulers.learning_rate_base import \
    LearningRateSchedulerBase


class MultiStepLearningRate(LearningRateSchedulerBase):
    """

    """

    def __init__(self, learning_rate, gamma=0.1, milestones=[]):
        self.learning_rate = learning_rate

    @classmethod
    def load_opt(cls, opt):
        return cls(
            opt.optimize_settings.learning_rate_scheduler.learning_rate,
            opt.optimize_settings.learning_rate_scheduler.gamma,
            opt.optimize_settings.learning_rate_scheduler.milestones
        )

    def get_learning_rate(self, epoch, step):
        return self.lr * self.gamma ** bisect_right(self.milestones, epoch)
