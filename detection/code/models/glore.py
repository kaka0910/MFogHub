###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################
from __future__ import division

import torch
import torch.nn as nn

from torch.nn.functional import upsample

from .base import BaseNet
from .fcn import FCNHead
from ..nn import BatchNorm1d, BatchNorm2d

class GloRe_Pyrimad(nn.Module):
    def __init__(self, in_channels=512, nodes=[8, 16, 32]):
        super(GloRe_Pyrimad, self).__init__()
        inter_channels = 256 #256
        cat_channels = 256

        # GloRe - Stage 5
        self.glore5 = GloRe_Unit_2D(in_channels, nodes[0])

        # GloRe - Stage 4
        self.glore4 = GloRe_Unit_2D(in_channels, nodes[1])

        # GloRe - Stage 3
        self.glore3 = GloRe_Unit_2D(in_channels, nodes[2])

        self.smooth = nn.Sequential(
            nn.Conv2d(in_channels*3, 512, 3, padding=1, bias=True),
            BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, c3, c4, c5):
        _, _, H, W = c3.size()

        x_32 = self.glore3(c3) + c3

        x_16 = self.glore4(c4) + c4

        x_8 = self.glore5(c5) + c5

        output = torch.cat([x_8, x_16, x_32], dim=1)
        output = self.smooth(output)

        return output

class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h


class GloRe_Unit(nn.Module):
    """
    Graph-based Global Reasoning Unit
    Parameter:
        'normalize' is not necessary if the input size is fixed
    """
    def __init__(self, num_in, num_mid, 
                 ConvNd=nn.Conv3d,
                 BatchNormNd=nn.BatchNorm3d,
                 normalize=False):
        super(GloRe_Unit, self).__init__()
        
        self.normalize = normalize
        self.num_s = 256
        self.num_n = num_mid

        # reduce dim
        self.conv_state = ConvNd(num_in, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = ConvNd(num_in, self.num_n, kernel_size=1)
        # ----------
        # reasoning via graph convolution
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # extend dimension
        self.conv_extend = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = BatchNormNd(num_in, eps=1e-04) # should be zero initialized


    def forward(self, x):
        '''
        :param x: (n, c, d, h, w)
        '''
        n = x.size(0)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state) + x_n_state

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])

        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        out = x + self.blocker(self.conv_extend(x_state))
        # out = self.blocker(self.conv_extend(x_state))

        return out

class GloRe_Unit_2D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=True):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(GloRe_Unit_2D, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv2d,
                                            BatchNormNd=BatchNorm2d,
                                            normalize=normalize)

