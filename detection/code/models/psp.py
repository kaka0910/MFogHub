###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################
from __future__ import division
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.functional import upsample
import torch.nn.functional as F

from .base import BaseNet
from .fcn import FCNHead
from .fph import FPH
from .dgr import CDGR
from .msgr import MDGR
from .glore import GloRe_Pyrimad
from .gcu import GCU_Pyrimad
from mmcv.ops import ModulatedDeformConv2dPack, DeformConv2dPack
from ..nn import BatchNorm1d, BatchNorm2d

def initialize_weights(models):
    """
    Initialize Model Weights
    """
    for name, module in models.named_modules():
        if 'gcn' in name:
            continue
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()

def initialize_weights_relu(models):
    """
    Initialize Model Weights
    """
    for name, module in models.named_modules():
        if 'gcn' in name:
            continue
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()

class GlobalPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(GlobalPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return F.interpolate(pool, (h,w), mode='bilinear', align_corners=True)

class FPH_(nn.Module):
    def __init__(self, in_channel=2048, out_channels=512, kernel_size=3, padding=1, dp=0.05, norm_layer=nn.BatchNorm2d):
        super(FPH_, self).__init__()
        c5 = in_channel       # 2048
        c4 = in_channel // 2  # 1024
        c3 = in_channel // 4  # 512

        # Head3 - Stage3
        self.head3 = nn.Sequential(
            nn.Dropout2d(dp, False),
            nn.Conv2d(c3, out_channels, kernel_size, padding=padding, bias=False),
            norm_layer(out_channels),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
        )
        self.head4 = nn.Sequential(
            nn.Dropout2d(dp, False),
            nn.Conv2d(c4, out_channels, kernel_size, padding=padding, bias=False),
            norm_layer(out_channels),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
        )
        self.head5 = nn.Sequential(
            nn.Dropout2d(dp, False),
            nn.Conv2d(c5, out_channels, kernel_size, padding=padding, bias=False),
            norm_layer(out_channels),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, c3, c4, c5):
        f3 = self.head3(c3)
        f4 = self.head4(c4)
        f5 = self.head5(c5)
        return f3, f4, f5

class PSPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PSPHead, self).__init__()
        inter_channels = in_channels // 4 # 512

        self.fph = FPH_(in_channels, 512, 3, 1)

        self.pyrimad_decoder = CDGR(inter_channels, [8,16,32], norm_layer) # [8,16,32] / [12,24,48]

        self.detail = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels//2, 1, padding=0, bias=False),
            norm_layer(inter_channels//2),
            nn.LeakyReLU(inplace=True)
        )
        self.gap = GlobalPooling(in_channels, inter_channels//2)

        self.conv6 = nn.Sequential(
            ModulatedDeformConv2dPack(inter_channels*2, # + inter_channels//2,
                                    inter_channels, 
                                    kernel_size=3, 
                                    padding=1, 
                                    deform_groups=4, 
                                    bias=False),
            norm_layer(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1)
        )

    def forward(self, f5, f4, f3):
        # GAP
        gap = self.gap(f5)

        f3, f4, f5 = self.fph(f3, f4, f5)
        feat = self.pyrimad_decoder(f3, f4, f5)

        # feat = torch.cat([feat, gap], dim=1)

        f3 = self.detail(f3)
        feat = torch.cat([feat, f3, gap], dim=1)

        # f5 = self.detail(f5)
        # feat = torch.cat([feat, f5], dim=1)

        feat = self.conv6(feat)

        return feat

class PSP(BaseNet):
    def __init__(self, nclass, backbone, criterion_seg=None, criterion_aux=None, aux=True, se_loss=False, norm_layer=BatchNorm2d, **kwargs):
        super(PSP, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = PSPHead(2048, nclass, norm_layer, self._up_kwargs)
        self.criterion_seg = criterion_seg
        self.criterion_aux = criterion_aux
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x, y=None):
        """
        (pcontext)
        Input:
            x: B, 3, 520, 520
        Intermedia:
            c3: B, 1024, 33, 33
            c4: B, 2048, 65, 65
        Output:
            out: B, 59, 520, 520
            out_aux: B, 59, 520, 520
        """
        _, _, h, w = x.size()
        _, c2, c3, c4 = self.base_forward(x)
        del _

        x = self.head(c4, c3, c2)
        x = F.interpolate(x, (h,w), **self._up_kwargs)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h,w), **self._up_kwargs)

        if self.training:
            # aux = self.auxlayer(c3)
            # aux = F.interpolate(aux, (h, w), **self._up_kwargs)
            main_loss = self.criterion_seg(x, y)
            aux_loss = self.criterion_aux(auxout, y)
            return x.max(1)[1], main_loss, aux_loss
        else: 
            # return x
            output = []
            output.append(x)
            output.append(auxout)
            return tuple(output)

# class PSP(BaseNet):
#     def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=BatchNorm2d, **kwargs):
#         super(PSP, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
#         self.head = PSPHead(2048, nclass, norm_layer, self._up_kwargs)
#         if aux:
#             self.auxlayer = FCNHead(1024, nclass, norm_layer)
#             # initialize_weights(self.auxlayer)

#     def forward(self, x):
#         """
#         (pcontext)
#         Input:
#             x: B, 3, 520, 520
#         Intermedia:
#             c3: B, 1024, 33, 33
#             c4: B, 2048, 65, 65
#         Output:
#             out: B, 59, 520, 520
#             out_aux: B, 59, 520, 520
#         """
#         _, _, h, w = x.size()
#         _, c2, c3, c4 = self.base_forward(x)
#         del _

#         outputs = []
#         x = self.head(c4, c3, c2)
#         x = upsample(x, (h,w), **self._up_kwargs)
#         outputs.append(x)

#         if self.aux:
#             auxout = self.auxlayer(c3)
#             auxout = upsample(auxout, (h,w), **self._up_kwargs)
#             outputs.append(auxout)
#         # outputs.append(vis)

#         return tuple(outputs)

def get_psp(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import datasets
    model = PSP(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)

    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('psp_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_psp_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_psp_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_psp('ade20k', 'resnet50', pretrained, root=root, **kwargs)

