import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.ops import DeformConv2d as DeformConv

from dcnet.encoders.encoder_base import EncoderBase
from dcnet.modules.dual_graph_convolution import DualGraphConv
from dcnet.modules.ghostnet import GhostBottleneck, GhostModule
from dcnet.modules.glore import GloReUnit
from dcnet.modules.linear_attention import (ChannelLinearAttention,
                                            PositionLinearAttention)

__all__ = ["ResNet", "resnet18", "resnet34",
           "resnet50", "resnet101", "resnet152"]

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        """

        """
        super(BasicBlock, self).__init__()

        self.stride = stride
        self.downsample = downsample
        self.with_dcn = dcn is not None

        self.conv1 = self.conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.with_modulated_dcn = False

        if self.with_dcn:
            fallback_on_stride = dcn.get("fallback_on_stride", False)
            self.with_modulated_dcn = dcn.get("modulated", False)

        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                padding=1,
                bias=False
            )
        else:
            deformable_groups = dcn.get("deformable_groups", 1)
            if not self.with_modulated_dcn:
                offset_channels = 18
                self.conv2_offset = nn.Conv2d(
                    planes,
                    deformable_groups * offset_channels,
                    kernel_size=3,
                    padding=1
                )

                self.conv2 = DeformConv(
                    planes,
                    planes,
                    kernel_size=3,
                    padding=1,
                    bias=False
                )
            else:
                offset_channels = 27
                self.conv2_offset = nn.Conv2d(
                    planes,
                    deformable_groups * offset_channels,
                    kernel_size=3,
                    padding=1
                )

                self.conv2 = ModulatedDeformConv(
                    planes,
                    planes,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    modulation=True
                )

    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        return conv

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)

        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class BottleneckWithGhostModule(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(BottleneckWithGhostModule, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # Replace convolution with ghost module
        self.ghost1 = GhostModule(inp=inplanes, oup=planes, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(planes)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get("fallback_on_stride", False)
            self.with_modulated_dcn = dcn.get("modulated", False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
        else:
            deformable_groups = dcn.get("deformable_groups", 1)

            if not self.with_modulated_dcn:
                from torchvision.ops import DeformConv2d as DeformConv
                conv_op = DeformConv
                offset_channels = 18
                self.conv2_offset = nn.Conv2d(
                    planes, deformable_groups * offset_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1)
                self.conv2 = conv_op(
                    planes, planes, kernel_size=3, padding=1, stride=stride,
                    # deformable_groups=deformable_groups,
                    bias=False)
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
                self.conv2_offset = nn.Conv2d(
                    planes, deformable_groups * offset_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1)
                self.conv2 = conv_op(
                    planes, planes, kernel_size=3, padding=1,
                    stride=stride,
                    # deformable_groups=deformable_groups,
                    bias=False,
                    modulation=True)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.ghost3 = GhostModule(
            inp=planes, oup=planes * 4, kernel_size=1, relu=True)

        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dcn = dcn
        self.with_dcn = dcn is not None

    def forward(self, x):
        residual = x
        out = self.ghost1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            # out = self.conv2(out, offset, mask)
            out = self.conv2(out)

        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.ghost3(out)

        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(Bottleneck, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get("fallback_on_stride", False)
            self.with_modulated_dcn = dcn.get("modulated", False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
        else:
            deformable_groups = dcn.get("deformable_groups", 1)

            if not self.with_modulated_dcn:
                from torchvision.ops import DeformConv2d as DeformConv
                conv_op = DeformConv
                offset_channels = 18
                self.conv2_offset = nn.Conv2d(
                    planes, deformable_groups * offset_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1)
                self.conv2 = conv_op(
                    planes, planes, kernel_size=3, padding=1, stride=stride,
                    # deformable_groups=deformable_groups,
                    bias=False)
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27

                self.conv2_offset = nn.Conv2d(
                    planes,
                    deformable_groups * offset_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1
                )

                self.conv2 = conv_op(
                    planes,
                    planes,
                    stride=stride,
                    kernel_size=3,
                    padding=1,
                    deformable_groups=deformable_groups,
                    bias=False
                )

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dcn = dcn
        self.with_dcn = dcn is not None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            out = self.conv2(out, offset, mask)
            # out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class DeformResNet(nn.Module):
    def __init__(self,
                 input_channel,
                 block,
                 layers,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 linear_attention=False,
                 dual_graph_conv=False):
        """

        """
        super(DeformResNet, self).__init__()
        self.dcn = dcn
        self.inplanes = 64
        self.stage_with_dcn = stage_with_dcn
        self.linear_attention = linear_attention
        self.dual_graph_conv = dual_graph_conv

        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_channel, 64, kernel_size=7,
                              stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dcn=dcn)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dcn=dcn)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dcn=dcn)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        if self.dual_graph_conv:
            self.dual_graph = GloReUnit(
                512 * block.expansion, 512, is_normalize=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.dcn is not None:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, BasicBlock) or isinstance(m, BottleneckWithGhostModule) or isinstance(m, GhostBottleneck):
                    if hasattr(m, "conv2_offset"):
                        self.constant_init(m.conv2_offset, 0)

    def constant_init(self, module, constant, bias=0):
        nn.init.constant_(module.weight, constant)
        if hasattr(module, "bias"):
            nn.init.constant_(module.bias, bias)

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):
        if block == GhostBottleneck:
            layers = []
            layers.append(
                block(
                    in_chs=self.inplanes, mid_chs=planes * block.expansion, out_chs=planes, dw_kernel_size=3,
                    stride=stride, act_layer=nn.ReLU, se_ratio=0.
                )
            )

            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(in_chs=self.inplanes,
                                    mid_chs=planes * block.expansion, out_chs=planes))

            return nn.Sequential(*layers)

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                dcn=dcn
            )
        )

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))

        if self.linear_attention:
            layers.append(PositionLinearAttention(planes * block.expansion))
            # layers.append(ChannelLinearAttention())

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        if self.dual_graph_conv:
            x5 = self.dual_graph(x5)

        return (x2, x3, x4, x5)


class DeformResnetEncoder(EncoderBase):
    """ An encoder for image embedding
    """

    def __init__(self, input_channel, linear_attention, dual_graph_conv, block_type, modulated, **kwargs):
        super(DeformResnetEncoder, self).__init__()

        block_dict = {"resnet_basic_block": BasicBlock,
                      "resnet_bottle_neck": Bottleneck,
                      "resnet_bottle_neck_ft_ghost_module": BottleneckWithGhostModule,
                      "ghost_bottle_neck": GhostBottleneck}

        self.encoder = DeformResNet(
            input_channel,
            block_dict[block_type],
            layers=[3, 4, 6, 3],
            dcn=dict(
                modulated=modulated,
                deformable_groups=1,
                fallback_on_stride=False
            ),
            stage_with_dcn=[False, True, True, True],
            linear_attention=linear_attention,
            dual_graph_conv=dual_graph_conv
        )
        # if kwargs.get("pretrained", True):
        #     self.encoder.load_state_dict(
        #         model_zoo.load_url(model_urls["resnet50"]), strict=False)

    @classmethod
    def load_opt(cls, opt):
        # import pdb; pdb.set_trace()
        if "dual_graph_conv" not in opt.network.encoder.keys():
            opt.network.encoder.dual_graph_conv = False
        if "modulated" not in opt.network.encoder.keys():
            opt.network.encoder.modulated = False
        return cls(
            opt.network.encoder.input_channel,
            opt.network.encoder.linear_attention,
            opt.network.encoder.dual_graph_conv,
            opt.network.encoder.block_type,
            opt.network.encoder.modulated
        )

    def forward(self, x):
        "See :obj:`.encoders.encoder_base.EncoderBase.forward()`"
        return self.encoder(x)
