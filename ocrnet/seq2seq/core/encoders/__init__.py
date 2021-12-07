"""Module defining encoders."""
from .encoder_base import EncoderBase
from .xception_backbone import XceptionEncoder

str2enc = {"xception": XceptionEncoder}

__all__ = ["EncoderBase", "XceptionEncoder", "str2enc"]
