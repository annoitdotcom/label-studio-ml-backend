import json
import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f'{num_params:,}'


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        if '[blank]' in dict_character:
            self.character = dict_character
        else:
            # dummy '[blank]' token for CTCLoss (index 0)
            self.character = ['[blank]'] + dict_character

    def encode(self, text, batch_max_length):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text).to(DEVICE), torch.IntTensor(length).to(DEVICE))

    def decode(self, logits, topk=2):
        """ convert text-index into text-label. 
        Select max probabilty (greedy decoding) then decode index to character
        Args:
            logits (torch.Tensor): output logits with size Batch * len seq * num classes
            delta (int): number of top k score to calculate confidence score
        Output:
            batch_text (lst): list of predicted strings
            batch_score (lst): list of confiddence scores  
        """

        index = 0
        batch_score = []
        batch_text = []
        batch_size = logits.size(0)

        probs_logits = nn.Softmax(dim=-1)(logits)
        probs_logits_np = probs_logits.data.cpu().numpy()[0]

        preds_size = torch.IntTensor([logits.size(1)] * batch_size)
        _, preds_index = logits.permute(1, 0, 2).max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1).data

        for l in preds_size.data:
            t = preds_index[index:index + l]

            chars = []
            scores = []
            for i in range(l):
                # removing repeated characters and blank.
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    chars.append(self.character[t[i]])
                    sorted_scores = sorted(probs_logits_np[i], reverse=True)
                    score = 1 - \
                        (np.sum(sorted_scores[1:topk])) / sorted_scores[0]
                    scores.append(score)

            text = ''.join(chars)
            batch_text.append(text)
            batch_score.append(scores)
            index += l

        return (batch_text, batch_score)


class NormalizePAD(object):
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        # img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            # Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
            Pad_img[:, :, w:] = 1
        Pad_img.sub_(0.5).div_(0.5)
        return Pad_img


class AlignCollate(object):
    def __init__(self, imgH=64, keep_ratio_with_pad=True):
        self.imgH = imgH
        self.max_width = 2000
        self.toTensor = transforms.ToTensor()
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def resize_img(self, img):
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = min(math.ceil(self.imgH * ratio), self.max_width)
        resized_image = cv2.resize(img,
                                   (resized_w, self.imgH), interpolation=cv2.INTER_CUBIC)
        return (resized_image, resized_w)

    def pad_img(self, img, max_size):
        img = self.toTensor(img)
        # img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        padding_img = torch.FloatTensor(*max_size).fill_(0)
        padding_img[:, :, :w] = img  # right pad
        if max_size[2] != w:  # add border Pad
            # padding_img[:, :, w:] = \
            #     img[:, :, w - 1].unsqueeze(2).expand(c, h, max_size[2] - w)
            padding_img[:, :, w:] = 1
        padding_img.sub_(0.5).div_(0.5)
        return padding_img

    def __call__(self, batch):
        outputs = list(map(lambda x: self.resize_img(x), batch))
        images, w_values = map(list, zip(*outputs))
        # transform = NormalizePAD((1, self.imgH, max(w_values)))

        if self.keep_ratio_with_pad:
            # image_tensors = list(map(transform, images))
            image_tensors = list(map(lambda img:
                                     self.pad_img(img, (1, self.imgH, max(w_values))), images))
            image_tensors = torch.cat([t.unsqueeze(0)
                                      for t in image_tensors], 0)
        else:
            image_tensors = list(map(self.toTensor, images))
            image_tensors = torch.cat([t.unsqueeze(0)
                                      for t in image_tensors], 0)
        return image_tensors
