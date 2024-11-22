###########################################################################
# Created by: Zhaoqing Wang
# Email: derrickwang005@gmail.com
# @BUPT 2020.11.18
###########################################################################
from __future__ import division

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNet
from .fcn import FCNHead
from ..utils import batch_pix_accuracy, batch_intersection_union

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class SELayer(nn.Module):
    def __init__(self, in_channels, reduct=4):
        super(SELayer, self).__init__()
        # self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduct, 1, bias=False),
            nn.PReLU(in_channels//reduct),
            nn.Conv2d(in_channels//reduct, in_channels, 1, bias=False),
            nn.Sigmoid()         
        )

    def forward(self, input):
        b, c, _, _ = input.size()
        del _
        # weight = self.pool(input).view(b, c, 1, 1)
        weight = self.se(input)
        return input * weight

class BasicBlock(nn.Module):
    """
    ResNet BasicBlock
    Groups: 4
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=False, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.bn1 = norm_layer(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1,
                               groups=1, padding=0, bias=False)
        self.bn2 = norm_layer(planes)
        self.se = SELayer(planes)
        if self.downsample:
            self.conv = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
                norm_layer(planes)
            )
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample:
            residual = self.conv(x)

        out += residual
        out = self.relu(out)

        return out

class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)

        #"""suppose a convolutional layer with g groups whose output has
        #g x n channels; we first reshape the output channel dimension
        #into (g, n)"""
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        #"""transposing and then flattening it back as the input of next layer."""
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x

class Block_cs(nn.Module):
    """
    Groups: 8
    """
    expansion = 1
    def __init__(self, inplanes, planes, outplanes, kernel_size, groups=1, downsample=False, norm_layer=nn.BatchNorm2d):
        super(Block_cs, self).__init__()
        self.downsample = downsample
        padding = 0 if kernel_size == 1 else 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn1 = norm_layer(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               groups=groups, padding=padding, bias=False)
        self.bn2 = norm_layer(planes)

        self.conv4 = nn.Conv2d(planes, outplanes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn4 = norm_layer(outplanes)

        if self.downsample:
            self.conv = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                norm_layer(outplanes)
            )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample:
            residual = self.conv(x)
        out += residual
        out = self.relu(out)

        return out

class FeatureFusion(nn.Module):
    def __init__(self, in_channels, reduct=8):
        super(FeatureFusion, self).__init__()
        self.gate1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduct, 1, bias=False),
            nn.PReLU(in_channels//reduct),
            nn.Conv2d(in_channels//reduct, in_channels, 1, bias=False),
            nn.Sigmoid()         
        )
        self.gate2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduct, 1, bias=False),
            nn.PReLU(in_channels//reduct),
            nn.Conv2d(in_channels//reduct, in_channels, 1, bias=False),
            nn.Sigmoid()         
        )
        self.gate4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduct, 1, bias=False),
            nn.PReLU(in_channels//reduct),
            nn.Conv2d(in_channels//reduct, in_channels, 1, bias=False),
            nn.Sigmoid()         
        )

    def forward(self, x1, x2, x4):
        coef_1 = self.gate1(x1).unsqueeze(-1)
        coef_2 = self.gate2(x2).unsqueeze(-1)
        coef_4 = self.gate4(x4).unsqueeze(-1)
        coef = F.softmax(torch.cat([coef_1, coef_2, coef_4], dim=-1), dim=-1)
        # print(coef[0, 0, 0, 0, :])
        # quit()
        x1 = x1 * coef[..., 0]
        x2 = x2 * coef[..., 1]
        x4 = x4 * coef[..., 2]
        return x1 + x2 + x4

class VECNet_v4(nn.Module):
    BASE_WIDTH=64
    def __init__(self, nclass, criterion_seg=None, criterion_aux=None, aux=False, norm_layer=nn.BatchNorm2d):
        super(VECNet_v4, self).__init__()
        self.nclass = nclass
        self.base_size = 256
        self.crop_size = 256
        self._up_kwargs = up_kwargs
        self.aux = aux
        self.stage_num = [3, 3] # [2, 2, 2, 2]
        
        # Input layer
        self.input = nn.Sequential(
            nn.Conv2d(16, self.BASE_WIDTH, 1, bias=False),
            norm_layer(self.BASE_WIDTH),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.BASE_WIDTH, self.BASE_WIDTH * 2, 1, bias=False),
            norm_layer(self.BASE_WIDTH * 2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.BASE_WIDTH * 2, self.BASE_WIDTH * 2, 1, bias=False),
            norm_layer(self.BASE_WIDTH * 2),
            nn.LeakyReLU(inplace=True),
        )
        # Encoder
        self.layer1 = self._make_layer(BasicBlock, self.BASE_WIDTH * 2, self.BASE_WIDTH * 4, self.stage_num[0], True)

        # Center Block
        self.b1x = nn.Sequential(
            Block_cs(self.BASE_WIDTH * 4, self.BASE_WIDTH * 8, self.BASE_WIDTH * 8, 1, downsample=True),
            Block_cs(self.BASE_WIDTH * 8, self.BASE_WIDTH * 8, self.BASE_WIDTH * 8, 1),
            Block_cs(self.BASE_WIDTH * 8, self.BASE_WIDTH * 8, self.BASE_WIDTH * 4, 1, downsample=True),
        )
        self.b2x = nn.Sequential(
                nn.Conv2d(self.BASE_WIDTH * 4, self.BASE_WIDTH * 8, 
                    kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(self.BASE_WIDTH * 8),
                nn.LeakyReLU(inplace=True),
                Block_cs(self.BASE_WIDTH * 8, self.BASE_WIDTH * 8, self.BASE_WIDTH * 8, 3, downsample=False),
                Block_cs(self.BASE_WIDTH * 8, self.BASE_WIDTH * 8, self.BASE_WIDTH * 4, 3, downsample=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(self.BASE_WIDTH * 4, self.BASE_WIDTH * 4, 
                    kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(self.BASE_WIDTH * 4),
                nn.LeakyReLU(inplace=True),
        )
        self.b4x = nn.Sequential(
                nn.Conv2d(self.BASE_WIDTH * 4, self.BASE_WIDTH * 8, 
                    kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(self.BASE_WIDTH * 8),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.BASE_WIDTH * 8, self.BASE_WIDTH * 8, 
                    kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(self.BASE_WIDTH * 8),
                nn.LeakyReLU(inplace=True),
                Block_cs(self.BASE_WIDTH * 8, self.BASE_WIDTH * 8, self.BASE_WIDTH * 8, 3, downsample=False),
                Block_cs(self.BASE_WIDTH * 8, self.BASE_WIDTH * 8, self.BASE_WIDTH * 4, 3, downsample=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(self.BASE_WIDTH * 4, self.BASE_WIDTH * 4, 
                    kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(self.BASE_WIDTH * 4),
                nn.LeakyReLU(inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(self.BASE_WIDTH * 4, self.BASE_WIDTH * 4, 
                    kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(self.BASE_WIDTH * 4),
                nn.LeakyReLU(inplace=True),
        )
        self.fuse1 = FeatureFusion(self.BASE_WIDTH * 4)
        # self.fuse2 = nn.Sequential(
        #     nn.Conv2d(self.BASE_WIDTH * 8, self.BASE_WIDTH * 4, 1, bias=False),
        #     norm_layer(self.BASE_WIDTH * 4),
        #     nn.LeakyReLU(inplace=True),
        # )
        # Decoder
        self.layer4 = self._make_layer(BasicBlock, self.BASE_WIDTH * 4, self.BASE_WIDTH * 4, self.stage_num[1], False)
        # Classifier
        self.output = nn.Sequential(
            nn.Conv2d(self.BASE_WIDTH * 4, self.BASE_WIDTH * 2, 1, bias=False),
            norm_layer(self.BASE_WIDTH * 2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.05, False),
            nn.Conv2d(self.BASE_WIDTH * 2, self.BASE_WIDTH * 2, 1, padding=0, bias=False),
            norm_layer(self.BASE_WIDTH * 2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.BASE_WIDTH * 2, self.nclass, 1)
        )
        # Criterion
        self.criterion_seg = criterion_seg
        self.criterion_aux = criterion_aux
        if self.aux:
            self.auxlayer = FCNHead(self.BASE_WIDTH * 8, nclass, norm_layer)
        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_channels, out_channels, num_blocks, downsample):
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        layers = [block(inplanes=in_channels, planes=out_channels, downsample=downsample)]
        for _ in range(num_blocks - 1):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        _, _, h, w = x.size()
        del _
        x = self.input(x)
        # Encoder
        feat1 = self.layer1(x)
        # Centre 
        feat_x1 = self.b1x(feat1)
        feat_x2 = self.b2x(feat1)
        feat_x4 = self.b4x(feat1)
        feat2 = self.fuse1(feat_x1, feat_x2, feat_x4)
        # feat2 = self.fuse2(torch.cat([feat1, feat2], dim=1))
        # Decoder
        feat3 = self.layer4(feat2)
        # Classifier
        output = self.output(feat3)

        if self.training:
            main_loss = self.criterion_seg(output, y)
            if self.aux:
                auxout = self.auxlayer(feat2)
                aux_loss = self.criterion_aux(auxout, y)
                return output.max(1)[1], main_loss, aux_loss
            return output.max(1)[1], main_loss
        else: 
            outputs = []
            outputs.append(output)
            return tuple(outputs)

    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union

def get_vec_v4(dataset='pascal_voc', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = VECNet_v4(datasets[dataset.lower()].NUM_CLASS, **kwargs)
    return model
