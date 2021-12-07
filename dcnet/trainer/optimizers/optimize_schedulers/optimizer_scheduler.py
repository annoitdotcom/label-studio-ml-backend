import torch

from dcnet.dataloader.config import Configurable, State


class OptimizerScheduler():
    optimizer = State()
    optimizer_args = State(default={})
    learning_rate = State(autoload=False)

    def __init__(self, opt):
        self.opt = opt
        self.learning_rate = self.opt.optimize_settings.learning_rate_scheduler.learning_rate
        self.optimizer = self.opt.optimize_settings.optimize_scheduler.type

        self.optimizer_args = dict(
            self.opt.optimize_settings.optimize_scheduler)
        self.optimizer_args = {
            k: v for (k, v) in self.optimizer_args.items() if k != "type"}
        self.optimizer_args.update({"lr": self.learning_rate})

    def create_optimizer(self, parameters):
        optimizer = getattr(torch.optim, self.optimizer)(
            parameters, **self.optimizer_args)
        if hasattr(self.learning_rate, "prepare"):
            self.learning_rate.prepare(optimizer)
        return optimizer
