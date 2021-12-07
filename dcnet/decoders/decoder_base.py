import torch
import torch.nn as nn


class DecoderBase(nn.Module):
    """ Base decoder class. Specifies the interface used by different decoder types """

    def __init__(self):
        super(DecoderBase, self).__init__()

    @classmethod
    def from_opt(cls, opt):
        raise NotImplementedError

    def count_parameters(self):
        num_params = sum(p.numel()
                         for p in self.parameters() if p.requires_grad)
        return f'{num_params:,}'

    def forward(self, x):
        """
        Args:
            x (:obj:`LongTensor`):
               padded sequences of sparse indices `[batch x src_len x nfeat]`
        Returns:
            (obj:`FloatTensor`):
        """
        raise NotImplementedError
