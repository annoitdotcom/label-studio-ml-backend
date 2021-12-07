import logging

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from .decoders import str2dec
from .encoders import str2enc
from .modules.luong_attention import Attention
from .modules.transformation import TPS_SpatialTransformerNetwork

logging.basicConfig(level=logging.DEBUG)


class SequenceModel(nn.Module):
    """ Initialize sequence to sequence modeling """

    def __init__(self, opt):
        super(SequenceModel, self).__init__()
        self.opt = opt
        self.stages = {'trans': self.opt.transformation, 'feat': self.opt.feature_extraction,
                       'seq': self.opt.sequence_modeling, 'pred': self.opt.prediction,
                       'pool': self.opt.adaptive_pooling}

        # Tranformation layers
        if self.opt.transformation != None:
            self.transformation = TPS_SpatialTransformerNetwork(
                F=self.opt.num_fiducial, I_size=(
                    self.opt.img_height, self.opt.img_width),
                I_r_size=(self.opt.img_height, self.opt.img_width), I_channel_num=self.opt.input_channel)

        # Feature extraction layers
        try:
            self.feature_extraction = str2enc[self.opt.feature_extraction].load_opt(
                self.opt)
            # int(imgH/16-1) * 512
            self.feature_extraction_output = self.opt.output_channel
            logging.debug("{0} num parameters: {1}".format(
                self.feature_extraction.__class__.__name__, self.feature_extraction.count_parameters()))
        except:
            raise Exception("Given encoder is only supported: {0}, but found: {1}".format(
                str2enc.keys(), self.opt.feature_extraction))

        # Last pooling layers
        if self.opt.adaptive_pooling != None:
            self.adaptive_pool = nn.AdaptiveAvgPool2d(
                (None, 1))  # Transform final (imgH/16-1) -> 1

        # Sequential decoder layers
        try:
            self.sequential_decoder = str2dec[self.opt.sequence_modeling].load_opt(
                self.opt)
            self.sequential_output_size = self.sequential_decoder.hidden_size
            logging.debug("{0} num parameters: {1}".format(
                self.sequential_decoder.__class__.__name__, self.sequential_decoder.count_parameters()))
        except:
            raise Exception("Given encoder is only supported: {0}, but found: {1}".format(
                str2dec.keys(), self.opt.sequence_modeling))

        # Prediction layers
        if self.opt.prediction == 'attn':
            self.prediction = Attention(
                self.sequential_output_size, self.opt.hidden_size, self.opt.num_class)
        elif (self.opt.prediction == 'ctc' or self.opt.prediction == 'ace'):
            self.prediction = nn.Linear(
                self.sequential_output_size, self.opt.num_class)
        else:
            raise Exception("Prediction is neither CTC, Attention or ACE")

    def forward(self, _input, _text=None, is_train=True):
        """ Transformation stage """
        if not self.stages['trans'] == None:
            _input = self.transformation(_input)

        """ Feature extraction stage """
        visual_feature = self.feature_extraction(_input)
        if self.stages['pool']:
            visual_feature = self.adaptive_pool(visual_feature.permute(
                0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
            visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        # Input: Batch x Steps x Channels
        # Output: Batch x Steps x Channels
        if self.stages['seq'] == None:
            # for convenience. this is NOT contextually modeled by BiLSTM
            contextual_feature = visual_feature
        else:
            contextual_feature = self.sequential_decoder(visual_feature)

        """ Prediction stage """
        if self.stages['pred'] == 'ctc' or self.stages['pred'] == 'ace':
            prediction = self.prediction(contextual_feature.contiguous())
        elif self.stages['pred'] == 'attn':
            prediction = self.prediction(contextual_feature.contiguous(),
                                         _text, is_train, batch_max_length=self.opt.batch_max_length)
        return prediction
