import logging
import os
import sys

import torch
import torch.nn as nn

from dcnet.decoders import str2dec
from dcnet.encoders import str2enc

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class NetworkBuilder(nn.Module):
    def __init__(self, opt, criterion=None, device="cpu"):
        """ Constructing differential contour text detection network
        Args:
            opt (dict): a dict of model"s configs
        """
        super(NetworkBuilder, self).__init__()
        self.opt = opt
        self.device = device
        self.to(self.device)

        # Encoder network
        try:
            self.encoder = str2enc[self.opt.network.encoder.type].load_opt(
                self.opt)
            if self.opt.optimize_settings.debug:
                logging.debug("{0} num parameters: {1}".format(
                    self.encoder.__class__.__name__, self.encoder.count_parameters()))
        except:
            raise Exception("Given encoder is only supported: {0}, but found: {1}".format(
                str2enc.keys(), self.opt.network.encoder.type))

        # Decoder network
        try:
            self.decoder = str2dec[self.opt.network.decoder.type].load_opt(
                self.opt)
            if self.opt.optimize_settings.debug:
                logging.debug("{0} num parameters: {1}".format(
                    self.decoder.__class__.__name__, self.decoder.count_parameters()))
        except:
            raise Exception("Given decoder is only supported: {0}, but found: {1}".format(
                str2dec.keys(), self.opt.network.decoder.type))

        self.criterion = criterion

    def forward(self, _input, is_training=False, *args, **kwargs):
        if isinstance(_input, dict):
            data = _input["image"].to(self.device)
        else:
            data = _input.to(self.device)

        data = data.float()
        output = self.decoder(self.encoder(data), *args, **kwargs)
        if is_training:
            if torch.isnan(output["thresh"]).any():
                return None
            for key, value in _input.items():
                if value is not None:
                    if hasattr(value, "to"):
                        _input[key] = value.to(self.device)
            loss_with_metrics = self.criterion(output, _input)
            loss, metrics = loss_with_metrics
            return loss, output, metrics

        return output
