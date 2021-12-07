import os


class LearningRateSchedulerBase(object):
    """ Base learning rate scheduler
    """

    def __init__(self):
        super(LearningRateSchedulerBase, self).__init__()

    @classmethod
    def load_opt(cls, opt):
        raise NotImplementedError

    def get_learning_rate(self, epoch, step=None):
        raise NotImplementedError
