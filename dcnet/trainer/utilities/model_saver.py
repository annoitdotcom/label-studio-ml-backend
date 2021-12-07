import os

import torch

from .signal_monitor import SignalMonitor


class ModelSaver(object):
    """

    """

    def __init__(self, opt):
        self.opt = opt

        # BUG: signal path should not be global
        self.monitor = SignalMonitor(self.opt.logging.signal_file_name)

    def save_model(self, model, config, epoch=None, step=None):
        if isinstance(model, dict):
            for name, net in model.items():
                checkpoint_name = self.make_checkpoint_name(name, epoch, step)
                self.save_checkpoint(net, config, checkpoint_name)
        else:
            checkpoint_name = self.make_checkpoint_name('model', epoch, step)
            self.save_checkpoint(model, config, checkpoint_name)

    def save_model_with_loss(self, model, config, epoch=None, step=None, loss=None):
        if isinstance(model, dict):
            for name, net in model.items():
                checkpoint_name = '{}_epoch_{}_minibatch_{}_{}'.format(
                    name, epoch, step, loss)
                self.save_checkpoint(net, config, checkpoint_name)
        else:
            checkpoint_name = '{}_epoch_{}_minibatch_{}_{}'.format(
                'model', epoch, step, loss)
            self.save_checkpoint(model, config, checkpoint_name)

    def save_checkpoint(self, net, config, name):
        output_path = os.path.join(
            self.opt.output_dir,
            self.opt.optimize_settings.model_dir_name
        )
        os.makedirs(output_path, exist_ok=True)
        torch.save({
            'state_dict': net.state_dict(),
            'config': config
        },
            os.path.join(output_path, name)
        )

    def make_checkpoint_name(self, name, epoch=None, step=None):
        if epoch is None or step is None:
            c_name = name + '_latest'
        else:
            c_name = '{}_epoch_{}_minibatch_{}'.format(name, epoch, step)
        return c_name

    def save_checkpoint_with_specific_path(self, net, config, output_path):
        torch.save({
            'state_dict': net.state_dict(),
            'config': config
        },
            output_path
        )
