
from collections import OrderedDict

import torch
import torch.nn as nn

from dcnet.decoders.decoder_base import DecoderBase


class HrDetector(nn.Module):
    def __init__(self,
                 inner_channels=256,
                 k=50,
                 bias=False,
                 adaptive=True,
                 smooth=False,
                 serial=False,
                 *args,
                 **kwargs):
        """
        Base decoder for segmenting instances
        Args:
            bias: Whether conv layers have bias or not.
            adaptive: Whether to use adaptive threshold training or not.
            smooth: If true, use bilinear instead of deconv.
            serial: If true, thresh prediction will combine segmentation result as input.
        """
        super(HrDetector, self).__init__()
        self.k = k
        self.serial = serial
        self.adaptive = adaptive

        # Binarize layer
        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
            nn.Sigmoid()
        )
        self.binarize.apply(self._weights_init)

        if self.adaptive:
            self.thresh = self._init_thresh(
                inner_channels, serial=serial, smooth=smooth, bias=bias
            )
            self.thresh.apply(self._weights_init)

    def _weights_init(self, m):
        """

        """
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        """

        """
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4,
                                inner_channels//4, smooth=smooth, bias=bias),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1,
                                smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        """

        """
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def _step_function(self, x, y):
        """

        """
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, fuse, gt=None, masks=None, training=False):
        binary = self.binarize(fuse)

        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary

        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                    (fuse, nn.functional.interpolate(binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self._step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result


class HrDecoder(DecoderBase):
    """ A decoder for segmenting instances """

    def __init__(self, opt, **kwargs):
        super(HrDecoder, self).__init__()
        self.decoder = HrDetector(
            inner_channels=opt.network.decoder.inner_channels,
            k=opt.network.decoder.k,
            bias=opt.network.decoder.bias,
            adaptive=opt.network.decoder.adaptive,
            smooth=opt.network.decoder.smooth,
            serial=opt.network.decoder.serial
        )

    @classmethod
    def load_opt(cls, opt):
        return cls(
            opt
        )

    def forward(self, x):
        "See :obj:`.encoders.encoder_base.DencoderBase.forward()`"
        return self.decoder(x)
