import torch.optim.lr_scheduler as lr_scheduler

from dcnet.trainer.optimizers.learning_rate_schedulers.learning_rate_base import \
    LearningRateSchedulerBase


class BuitlinLearningRate(LearningRateSchedulerBase):
    """

    """

    def __init__(self, learning_rate=0.001, klass="StepLR", *args, **kwargs):
        self.learning_rate = learning_rate
        self.klass = klass
        self.args = args
        self.kwargs = kwargs
        self.scheduler = None

    @classmethod
    def load_opt(cls, opt):
        return cls(
            opt.optimize_settings.learning_rate_scheduler.learning_rate,
            opt.optimize_settings.learning_rate_scheduler.klass,
            opt.optimize_settings.learning_rate_scheduler.args,
            opt.optimize_settings.learning_rate_scheduler.kwargs
        )

    def _get_scheduler(self, optimizer):
        self.scheduler = getattr(lr_scheduler, self.klass)(
            optimizer, *self.args, **self.kwargs)

    def get_learning_rate(self, epoch, step=None):
        if self.scheduler is None:
            raise "learning rate not ready (prepared with optimizer)"
        self.scheduler.last_epoch = epoch

        # Return value of gt_lr is a list,
        # where each element is the corresponding learning rate for a
        # paramater group.
        return self.scheduler.get_lr()[0]
