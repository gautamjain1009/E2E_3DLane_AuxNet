"""
Below code is modified from the official implementation of RESA: https://github.com/ZJULearning/resa/blob/main/models/decoder.py
"""

import torch
from torch import nn
import torch.nn.functional as F

from ..registry import HEADS

@HEADS.register_module
class PlainDecoder(nn.Module):
    def __init__(self, cfg):
        super(PlainDecoder, self).__init__()
        self.cfg = cfg

        # self.dropout = nn.Dropout2d(0.1)
        self.conv8 = nn.Conv2d(cfg.featuremap_out_channel, cfg.num_classes, 1)

    def forward(self, x):

        # x = self.dropout(x)
        x = self.conv8(x)

        ## F.interpolate and nn.Upsample are the same thing for upsampling but both classes are designed
        # for different use cases or one of them is depriciated.
        output = F.interpolate(x, size=[self.cfg.img_height,  self.cfg.img_width],
                           mode='bilinear', align_corners=False)

        return output 