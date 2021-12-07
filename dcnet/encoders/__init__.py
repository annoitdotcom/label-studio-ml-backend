"""Module defining encoders."""
from dcnet.encoders.deform_resnet_encoder import DeformResnetEncoder
from dcnet.encoders.effnetv2 import Effv2Encoder
from dcnet.encoders.encoder_base import EncoderBase
from dcnet.encoders.hrnet_encoder import HrnetEncoder
from dcnet.encoders.resnest_encoder import ResnestEncoder

str2enc = {
    "deform_resnet_encoder": DeformResnetEncoder,
    "resnest_encoder": ResnestEncoder,
    "hrnet_encoder": HrnetEncoder,
    "effv2_encoder": Effv2Encoder,
}

__all__ = ["EncoderBase", "Effv2Encoder", "HrnetEncoder",
           "ResnestEncoder", "DeformResnetEncoder", "str2enc"]
