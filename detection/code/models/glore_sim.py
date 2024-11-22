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
from ..nn import PyramidPooling

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def self_similarity(self,x):
        '''
        input: B, n_class, n_channel
        out: B, n_class, n_class
        '''
        sim = torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(x, p=2, dim=-1).transpose(-1, -2))
        sim = F.softmax(sim, dim=-1) #-1指的是方向
        return sim

    def forward(self, input):
        input = input.permute(0, 2, 1)
        adj = self.self_similarity(input)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        output = output
        if self.bias is not None:
            return (output + self.bias + input).permute(0, 2, 1)
        else:
            return output.permute(0, 2, 1)

class Global_Reason_Unit(nn.Module):
    """
    Reference:
       Chen, Yunpeng, et al. "Graph-Based Global Reasoning Networks" (https://arxiv.org/abs/1811.12814)
    """
    def __init__(self, in_channel, norm_layer, num_state=128, num_node=64):
        super(Global_Reason_Unit, self).__init__()
        """
           Graph G = (V, \sigma), |V| is the number of vertext, denoted as num_node, each vertex feature is d-dim vector, d is equal to num_state
        """
        self.num_state= num_state
        # generate Projection matrix and Reverse Projection Matrixs B
        self.conv_theta = nn.Conv2d(in_channel, num_node, kernel_size=1, stride=1, padding=0, bias=True)

        # Reduce Dim
        self.conv_phi =  nn.Conv2d(in_channel, num_state, kernel_size=1, stride=1, padding=0, bias=True)
        self.graph_conv = GraphConvolution(num_state, num_state)
        self.bn = norm_layer(in_channel)
        nn.init.constant_(self.bn.weight, 0)
        nn.init.constant_(self.bn.bias, 0)
        # Extend Dim
        self.extend_dim = nn.Conv2d(num_state, in_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        """
            input: x \in R^{B x C x H x W}
            output: x_new \in R^{B x C x H x W}
        """
        # generate B
        size= x.size()                                             # B x C x H x W
        B = self.conv_theta(x)                                     # B x |V| x H x W, where |V|= num_node
        size_B = B.size()
        B = B.reshape(size_B[0], size_B[1], -1)                    # Projection Matrix: B x |v| x L , (L= HxW)
        # Reduce_dim
        x_reduce = self.conv_phi(x)                                # B x C x H x W ---> B x C1 x H x W, where C1= num_state
        x_reduce = x_reduce.view(size[0], -1, size[2]*size[3]).permute(0,2,1)   # B x C1 x H x W -> B x C1 x L -> B  x L x C1

        V = torch.bmm(B, x_reduce).permute(0,2,1)             # B x |V| x C1  --> B x C1 x |V|
        V = torch.div(V, size[2]*size[3])                     # normalize V/HW
        new_V = self.graph_conv(V)                            # B x C1 x |V|

        D = B.view(size_B[0], -1, size_B[2]*size_B[3]).permute(0,2,1)     # Reverse Projection Matrix:        B x |V| x L -> B x L x |V|
        Y = torch.bmm(D, new_V.permute(0,2,1))                # B x L x C1
        Y = Y.permute(0,2,1)                                 # B x C1 x L
        Y = Y.reshape(size[0], self.num_state, size[2], -1)   # B x C1 x H x W
        Y = self.extend_dim(Y)                                # B x C x H x W
        Y = self.bn(Y)                                        # normalization
        out =  Y + x
        return out

class PSPHead(nn.Module):
    #def __init__(self, in_channels, out_channels, norm_layer=None):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PSPHead, self).__init__()
        inter_channels = in_channels //4
        # 3x3 Conv. 2048 -> 512
        self.conv5 = nn.Sequential( nn.Conv2d(in_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                            norm_layer(inter_channels),
                                            nn.ReLU(inplace=True))
        self.gloru= Global_Reason_Unit(in_channel= inter_channels, norm_layer=norm_layer, num_state=256, num_node=84)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))
    def forward(self, x):
        out = self.conv5(x)
        out = self.gloru(out)
        out = self.conv6(out)
        return out


class PSP(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(PSP, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = PSPHead(2048, nclass, norm_layer, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c4)
        x = upsample(x, (h,w), **self._up_kwargs)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = upsample(auxout, (h,w), **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


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

