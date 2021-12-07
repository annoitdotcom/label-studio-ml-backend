import torch
import torch.nn as nn
from torch.autograd import Variable

from .decoder_base import DecoderBase


class LSTM(nn.Module):
    def __init__(self, input_size, num_hidden, output_size,
                 dropout, num_layers, bidirectional=False):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, num_hidden,
                           num_layers=num_layers, dropout=dropout,
                           bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(num_hidden * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class LSTMDecoder(DecoderBase):
    """ A decoder based on Temporal convolutional neural network """

    def __init__(self, input_channels, num_hidden,
                 output_channels, dropout, num_layers, bidirectional):
        super(LSTMDecoder, self).__init__()

        self.rnn_decoder = LSTM(input_channels, num_hidden,
                                output_channels, dropout, num_layers, bidirectional)
        self.hidden_size = output_channels

    @classmethod
    def load_opt(cls, opt):
        return cls(
            opt.output_channel,
            opt.hidden_size,
            opt.hidden_size,
            opt.dropout_rate,
            opt.num_layers,
            opt.bidirectional
        )

    def forward(self, x):
        "See :obj:`.encoders.encoder_base.DencoderBase.forward()`"
        return self.rnn_decoder(x)
