from dcnet.trainer.optimizers.learning_rate_schedulers.learning_rate_base import \
    LearningRateSchedulerBase


class ConstantLearningRate(LearningRateSchedulerBase):
    """

    """

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @classmethod
    def load_opt(cls, opt):
        return cls(
            opt.optimize_settings.learning_rate_scheduler.learning_rate
        )

    def get_learning_rate(self, epoch, step):
        return self.learning_rate
