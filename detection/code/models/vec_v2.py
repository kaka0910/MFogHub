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

from .fcn import FCNHead
from ..utils import batch_pix_accuracy, batch_intersection_union
from mmcv.ops import CrissCrossAttention

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class SELayer(nn.Module):
    def __init__(self, in_channels, reduct=4):
        super(SELayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduct, 1, bias=False),
            nn.PReLU(in_channels//reduct),
            nn.Conv2d(in_channels//reduct, in_channels, 1, bias=False),
            nn.Sigmoid()         
        )

    def forward(self, input):
        b, c, _, _ = input.size()
        del _
        weight = self.pool(input).view(b, c, 1, 1)
        weight = self.se(weight)
        return input * weight

class SAM(nn.Module):
    def __init__(self, in_channels, reduct=4):
        super(SAM, self).__init__()
        self.se = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduct, 1, bias=False),
            nn.PReLU(in_channels//reduct),
            nn.Conv2d(in_channels//reduct, in_channels, 1, bias=False),
            nn.Sigmoid()         
        )

    def forward(self, input):
        weight = self.se(input)
        return input * weight

class SPG(nn.Module):
    def __init__(self, in_channels, nclass, norm_layer, criterion):
        super(SPG, self).__init__()
        self.aux_pred = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels//2, 1, bias=False),
            norm_layer(in_channels//2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels//2, nclass, 1)
        )
        self.att = nn.Sequential(
            nn.Conv2d(nclass, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.Sigmoid(),
        )
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.criterion = criterion
    
    def forward(self, x, y=None):
        feat = self.trans(x)
        coarse_pred = self.aux_pred(x)
        att_map = self.att(coarse_pred)
        output = feat * att_map + x
        output = self.conv(output)
        if self.training:
            coarse_pred = F.interpolate(coarse_pred, scale_factor=2, mode='bilinear', align_corners=False)
            loss = self.criterion(coarse_pred, y)
            return output, loss
        else:
            return output

class SPG_(nn.Module):
    def __init__(self, in_channels, nclass, norm_layer, criterion):
        super(SPG_, self).__init__()
        self.aux_pred = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels//2, 1, bias=False),
            norm_layer(in_channels//2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels//2, nclass, 1)
        )
        self.att = nn.Sequential(
            nn.Softmax(dim=1),
            nn.Conv2d(nclass, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.Sigmoid(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.criterion = criterion
    
    def forward(self, x, y=None):
        coarse_pred = self.aux_pred(x)
        att_map = self.att(coarse_pred)
        output = x * att_map
        output = self.conv(output)
        if self.training:
            coarse_pred = F.interpolate(coarse_pred, scale_factor=2, mode='bilinear', align_corners=False)
            loss = self.criterion(coarse_pred, y)
            return output, loss
        else:
            return output

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

## jpu
class JPU(nn.Module):
    def __init__(self, in_channels, conv='spb_conv', norm_layer=nn.BatchNorm2d, criterion=None):
        super(JPU, self).__init__()
        width = in_channels // 4
        assert conv == 'spb_conv' or conv == 'norm_conv', "Wrong Conv Type!"
        if conv == 'spb_conv':
            conv_layer = SeparableConv2d
        else: 
            conv_layer = nn.Conv2d
        
        self.dilation1 = nn.Sequential(conv_layer(in_channels, width, kernel_size=1, padding=0, dilation=1, bias=False), # utilize kernel_size=1
                                       norm_layer(width),
                                       nn.LeakyReLU(inplace=True),
                                    #    SAM(width, reduct=2)
                                       )
        self.spg1 = SPG_(width, 10, norm_layer, criterion)
        self.dilation2 = nn.Sequential(conv_layer(in_channels, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       nn.LeakyReLU(inplace=True),
                                    #    SAM(width, reduct=2)
                                       )
        self.spg2 = SPG_(width, 10, norm_layer, criterion)
        self.dilation3 = nn.Sequential(conv_layer(in_channels, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       nn.LeakyReLU(inplace=True),
                                    #    SAM(width, reduct=2)
                                       )
        self.spg3 = SPG_(width, 10, norm_layer, criterion)
        self.dilation4 = nn.Sequential(conv_layer(in_channels, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       nn.LeakyReLU(inplace=True),
                                    #    SAM(width, reduct=2)
                                       )
        self.spg4 = SPG_(width, 10, norm_layer, criterion)

        self.fuse = nn.Sequential(
            nn.Conv2d(4 * width, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, y=None):
        residual = feat
        f1 = self.dilation1(feat)
        f2 = self.dilation2(feat)
        f3 = self.dilation3(feat)
        f4 = self.dilation4(feat)
        if self.training:
            f1, l1 = self.spg1(f1, y)
            f2, l2 = self.spg2(f2, y)
            f3, l3 = self.spg3(f3, y)
            f4, l4 = self.spg4(f4, y)
            feat = torch.cat([f1, f2, f3, f4], dim=1)
            feat = self.fuse(feat) * self.gamma + residual
            loss = l1 + l2 + l3 + l4
            return feat, loss
        else:
            f1 = self.spg1(f1)
            f2 = self.spg2(f2)
            f3 = self.spg3(f3)
            f4 = self.spg4(f4)
            feat = torch.cat([f1, f2, f3, f4], dim=1)
            feat = self.fuse(feat) * self.gamma + residual
            return feat

class BasicBlock(nn.Module):
    """
    ResNet BasicBlock
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, groups=1, downsample=False, norm_layer=nn.BatchNorm2d, cam='se'):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.bn1 = norm_layer(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1,
                               groups=groups, padding=0, bias=False)
        self.bn2 = norm_layer(planes)
        self.se = SELayer(planes) if cam == 'se' else SAM(planes)
        # self.se = SAM(planes)
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

class VECNet(nn.Module):
    BASE_WIDTH=64
    def __init__(self, nclass, criterion_seg=None, criterion_aux=None, aux=False, norm_layer=nn.BatchNorm2d):
        super(VECNet, self).__init__()
        self.nclass = nclass
        self.base_size = 256
        self.crop_size = 256
        self._up_kwargs = up_kwargs
        self.aux = aux
        self.stage_num = [5, 6, 2, 2] # [2, 2, 2, 2]
        
        # Input layer 输入16通道
        self.input = nn.Sequential(
            nn.Conv2d(16, self.BASE_WIDTH, 1, bias=False),
            norm_layer(self.BASE_WIDTH),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.BASE_WIDTH, self.BASE_WIDTH * 2, 1, bias=False),
            norm_layer(self.BASE_WIDTH * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.BASE_WIDTH * 2, self.BASE_WIDTH * 4, 1, bias=False),
            norm_layer(self.BASE_WIDTH * 4),
            nn.LeakyReLU(inplace=True),
        )
        # Stage - 1
        self.layer1 = self._make_layer(BasicBlock, self.BASE_WIDTH * 4, self.BASE_WIDTH * 4, self.stage_num[0], False, 'se')
        # DownSample H:256 -> 128 C:256 -> 512
        self.down1 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, 
                    ceil_mode=True),
                # nn.AvgPool2d(kernel_size=2, stride=2, 
                #     ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.BASE_WIDTH * 4, self.BASE_WIDTH * 8, 
                    kernel_size=1, stride=1, padding=0, bias=False),
                # nn.Conv2d(self.BASE_WIDTH * 4, self.BASE_WIDTH * 8, 
                #     kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.BASE_WIDTH * 8),
                nn.LeakyReLU(inplace=True)
        )
        # Stage - 2
        self.layer2 = self._make_layer(BasicBlock, self.BASE_WIDTH * 8, self.BASE_WIDTH * 8, self.stage_num[1], False, 'se')

        # # CC Attention
        # self.attention = CCAtt(in_channels=self.BASE_WIDTH * 8)

        # JPU
        self.attention = JPU(in_channels=self.BASE_WIDTH * 8, criterion=criterion_aux)

        # Decoder1
        self.decoder1 = self._make_layer(BasicBlock, self.BASE_WIDTH * 8, self.BASE_WIDTH * 4, self.stage_num[2], True)
        
        # Upsample
        self.up1 = nn.Sequential(
            nn.Conv2d(self.BASE_WIDTH * 4, self.BASE_WIDTH * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.BASE_WIDTH * 4),
            nn.LeakyReLU(inplace=True)
        )
        
        # Decoder2
        self.decoder2 = self._make_layer(BasicBlock, self.BASE_WIDTH * 4, self.BASE_WIDTH * 4, self.stage_num[3], False)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(self.BASE_WIDTH * 4, self.BASE_WIDTH * 2, 1, bias=False),
            norm_layer(self.BASE_WIDTH * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.BASE_WIDTH * 2, self.BASE_WIDTH * 2, 1, bias=False),
            norm_layer(self.BASE_WIDTH * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.BASE_WIDTH * 2, self.nclass, 1)
        )
        
        # Aux Classifier
        if self.aux:
            self.aux_classifier = nn.Sequential(
            nn.Conv2d(self.BASE_WIDTH * 4, self.BASE_WIDTH * 2, 1, bias=False),
            norm_layer(self.BASE_WIDTH * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.BASE_WIDTH * 2, self.BASE_WIDTH * 2, 1, bias=False),
            norm_layer(self.BASE_WIDTH * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.BASE_WIDTH * 2, self.nclass, 1)
        )

        # Criterion
        self.criterion_seg = criterion_seg
        self.criterion_aux = criterion_aux

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_channels, out_channels, num_blocks, downsample, cam='se'):
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        layers = [block(inplanes=in_channels, planes=out_channels, downsample=downsample, cam=cam)]
        for _ in range(num_blocks - 1):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        # _, _, h, w = x.size()
        # del _
        x = self.input(x)

        feat1 = self.layer1(x)

        feat2 = self.down1(feat1)

        feat3 = self.layer2(feat2)

        if self.training:
            feat4, jpu_loss = self.attention(feat3, y)
        else:
            feat4 = self.attention(feat3)

        feat5 = self.decoder1(feat4)

        feat5 = F.interpolate(feat5, scale_factor=2, mode='bilinear', align_corners=False)
        feat5 = self.up1(feat5) + feat1

        feat6 = self.decoder2(feat5)

        output = self.classifier(feat6)

        if self.training:
            main_loss = self.criterion_seg(output, y)
            if self.aux:
                auxout = self.aux_classifier(feat3)
                aux_loss = self.criterion_aux(auxout, y)
                return output.max(1)[1].detach(), main_loss, jpu_loss, aux_loss
            return output.max(1)[1].detach(), main_loss, jpu_loss
        else: 
            outputs = []
            outputs.append(output.detach())
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

def get_vec_v2(dataset='pascal_voc', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = VECNet(datasets[dataset.lower()].NUM_CLASS, **kwargs)
    return model
