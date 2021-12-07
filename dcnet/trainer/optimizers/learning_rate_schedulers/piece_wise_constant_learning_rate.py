from dcnet.trainer.optimizers.learning_rate_schedulers.learning_rate_base import \
    LearningRateSchedulerBase


class PiecewiseConstantLearningRate(LearningRateSchedulerBase):
    """

    """

    def __init__(self, boundaries=[10000, 20000], values=[0.001, 0.0001, 0.00001]):
        self.boundaries = boundaries
        self.values = values

    @classmethod
    def load_opt(cls, opt):
        return cls(
            opt.optimize_settings.learning_rate_scheduler.boundaries,
            opt.optimize_settings.learning_rate_scheduler.values
        )

    def get_learning_rate(self, epoch, step):
        for boundary, value in zip(self.boundaries, self.values[:-1]):
            if step < boundary:
                return value
        return self.values[-1]
