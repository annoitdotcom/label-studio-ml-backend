from dcnet.trainer.optimizers.learning_rate_schedulers.learning_rate_base import \
    LearningRateSchedulerBase


class WarmupLearningRate(LearningRateSchedulerBase):
    """

    """

    def __init__(self, learning_rate, warmup_learning_rate=1e-5, steps=4000):
        self.learning_rate = learning_rate
        self.warmup_learning_rate = warmup_learning_rate
        self.steps = steps

    @classmethod
    def load_opt(cls, opt):
        return cls(
            opt.optimize_settings.learning_rate_scheduler.learning_rate,
            opt.optimize_settings.learning_rate_scheduler.warmup_learning_rate,
            opt.optimize_settings.learning_rate_scheduler.steps
        )

    def get_learning_rate(self, epoch, step):
        if epoch == 0 and step < self.steps:
            return self.warmup_learning_rate
        return self.learning_rate
