"""Module defining encoders."""
from .decoder_base import DecoderBase
from .lstm_backbone import LSTMDecoder
from .tcn_backbone import TCNDecoder

str2dec = {"lstm": LSTMDecoder, "tcn": TCNDecoder}

__all__ = ["DecoderBase", "LSTMDecoder", "TCNDecoder", "str2dec"]
