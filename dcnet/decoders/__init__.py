"""Module defining decoders."""
from dcnet.decoders.binarize_decoder import BinarizeDecoder
from dcnet.decoders.decoder_base import DecoderBase
from dcnet.decoders.hrnet_decoder import HrDecoder

str2dec = {
    "binarize_decoder": BinarizeDecoder,
    "hrnet_decoder": HrDecoder
}

__all__ = ["DecoderBase", "HrDecoder", "BinarizeDecoder", "str2dec"]
