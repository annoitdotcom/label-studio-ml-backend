import os

import torch


class Checkpoint():

    def __init__(self, opt, restore_path=None):
        self.opt = opt
        self.start_epoch = self.opt.optimize_settings.start_epoch
        self.start_iter = self.opt.optimize_settings.start_iter

        if restore_path:
            self.resume = restore_path
        elif self.opt.optimize_settings.restore_path:
            self.resume = self.opt.optimize_settings.restore_path
        else:
            self.resume = None

    def restore_model(self, model, device, logger):
        if self.resume is None:
            return

        if not os.path.exists(self.resume):
            logger.warning("Checkpoint not found: " + self.resume)
            return

        dict_config = dict()
        checkpoint = torch.load(self.resume, map_location=device)
        if 'config' in checkpoint.keys():
            dict_config = checkpoint['config']
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        logger.info("Resumed from " + self.resume)
        return dict_config

    def restore_counter(self):
        return self.start_epoch, self.start_iter
