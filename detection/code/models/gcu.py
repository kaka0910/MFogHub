###########################################################################
# Created by: Zhaoqing Wang
# Email: derrickwang005@gmail.com
# Copyright (c) 2020
###########################################################################
from __future__ import division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from ..nn import BatchNorm1d, BatchNorm2d

class GCU_Pyrimad(nn.Module):
    def __init__(self, in_channels=256, nodes=[8, 16, 32]):
        super(GCU_Pyrimad, self).__init__()
        # GCU - Stage 5
        self.conv8 = nn.Sequential(
            # nn.Dropout2d(0.05, False),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            # BatchNorm2d(256),
            # nn.ReLU(inplace=True)
        )
        self.proj_8 = Code_projection(in_channels=in_channels, D=in_channels, K=nodes[0])
        self.gcn_8 = GraphConvolution(in_channels, in_channels)
        self.ext8 = nn.Conv2d(256, 512, 1, padding=0, bias=False)

        # GloRe - Stage 4
        self.conv16 = nn.Sequential(
            # nn.Dropout2d(0.05, False),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            # BatchNorm2d(256),
            # nn.ReLU(inplace=True)
        )
        self.proj_16 = Code_projection(in_channels=in_channels, D=in_channels, K=nodes[1])
        self.gcn_16 = GraphConvolution(in_channels, in_channels)
        self.ext16 = nn.Conv2d(256, 512, 1, padding=0, bias=False)

        # GloRe - Stage 3
        self.conv32 = nn.Sequential(
            # nn.Dropout2d(0.05, False),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            # BatchNorm2d(256),
            # nn.ReLU(inplace=True)
        )
        self.proj_32 = Code_projection(in_channels=in_channels, D=in_channels, K=nodes[2])
        self.gcn_32 = GraphConvolution(in_channels, in_channels)
        self.ext32 = nn.Conv2d(256, 512, 1, padding=0, bias=False)

        self.smooth = nn.Sequential(
            nn.Conv2d(1536, 512, 3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def reproj(self, graph, Q, H, W):
        B, D, _ = graph.size()
        x = torch.matmul(graph, Q.permute(0, 2, 1)) # B, D, N
        x = x.reshape(B, D, H, W)
        return x

    def forward(self, c3, c4, c5):
        _, _, H, W = c3.size()

        graph_32, Q_32 = self.proj_32(self.conv32(c3))
        graph_32 = self.gcn_32(graph_32)
        x_32 = self.ext32(self.reproj(graph_32, Q_32, H, W)) + c3

        graph_16, Q_16 = self.proj_16(self.conv16(c4))
        graph_16 = self.gcn_16(graph_16)
        x_16 = self.ext16(self.reproj(graph_16, Q_16, H, W)) + c4

        graph_8, Q_8 = self.proj_8(self.conv8(c5))
        graph_8 = self.gcn_8(graph_8)
        x_8 = self.ext8(self.reproj(graph_8, Q_8, H, W)) + c5

        output = torch.cat([x_8, x_16, x_32], dim=1)
        output = self.smooth(output)

        return output


class Code_projection(nn.Module):
    def __init__(self, in_channels, D=512, K=32, use_channel_scale=True):
        super(Code_projection, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K  # K is the number of codewords, D is the feature channel of each codeword
        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)

        self.use_channel_scale= use_channel_scale
        if self.use_channel_scale:
            self.scale = nn.Parameter(torch.Tensor(D, K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        if self.use_channel_scale:
            self.scale.data.uniform_(-1, 0)

    def embedded_residual(self, x, codes):
        """
          x: B x D x H x W, N=HxW
          codes: B x D x K

          Z: B x D x K
          Q: B x N x K
          A: B, K, K
        """
        # residual: B x N x D x K
        residual = (x.unsqueeze(3) - codes) * self.scale #B, N, D, K
        Q = F.softmax(-(residual.pow(2).sum(2)/2), 2) # B, N, K
        Z_ = (((Q * residual.permute(2, 0, 1, 3)).sum(2))/Q.sum(1)).permute(1, 0, 2)     #B, D, K /B, K = B, D, K
        # Z = F.normalize(Z, p=2, dim=-1)
        Z = (Z_.permute(1, 0, 2)/torch.sqrt((Z_.pow(2).sum(1)))).permute(1, 0, 2) #B, D, K
        return Z, Q

    def forward(self, X):
        """
            X: B, 256, H, W
        """
        # assert X.dim() == 4
        # BxDxHxW
        B, D = X.size(0), X.size(1)
        X = X.view(B, D, -1).transpose(1, 2).contiguous()  # B x N x D, N=HxW
        # B x N x K
        Z, Q = self.embedded_residual(X, self.codewords.t())
        return Z, Q

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    *** Normalize adjcent matrix
    """

    def __init__(self, in_features, out_features, norm_layer=BatchNorm1d, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)

        self.relu = nn.ReLU(inplace=True)
        self.bn = norm_layer(out_features)
        nn.init.constant_(self.bn.weight, 0)
        nn.init.constant_(self.bn.bias, 0)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def norm_adj(self, input):
        """
            input: B, C, K
        """
        adj = torch.matmul(input.permute(0,2,1), input)
        return adj
    
    def forward(self, input):
        adj = self.norm_adj(input)
        support = torch.matmul(self.weight, input)
        output = torch.matmul(support, adj)

        if self.bias is not None:
            return self.bn(self.relu(output + self.bias)) 
        else:
            return self.bn(self.relu(output))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
