import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from ..nn import BatchNorm1d, BatchNorm2d
from .dgr import A2_Projection, GraphConvolution, GraphConvolutionTopK

class MDGR(nn.Module):
    def __init__(self, in_channels=512, nodes=[8, 16, 32]):
        super(MDGR, self).__init__()
        inter_channels = 256 #256
        # Double Attention Proj - 5
        self.proj_8 = A2_Projection(in_channels, nodes[0])

        # Double Attention Proj - 4
        self.proj_16 = A2_Projection(in_channels, nodes[1])

        # Double Attention Proj - 3
        self.proj_32 = A2_Projection(in_channels, nodes[2])

        # GCN 
        self.gamma8 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.gamma16 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.gamma32 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.gcn_8 = GraphConvolution(inter_channels, inter_channels)
        self.gcn_16 = GraphConvolution(inter_channels, inter_channels)
        self.gcn_32 = GraphConvolution(inter_channels, inter_channels)

        # Multi-scale in graph space
        # Graph 32
        # self.gcn_1632 = CrossGraphConvolution(inter_channels, inter_channels//2, L2=True)
        # self.gcn_832 = CrossGraphConvolution(inter_channels, inter_channels//2, L2=True)
        self.gcn816_32 = CrossGraphSample(L2=True)
        # self.gcn_832 = CrossGraphSample(inter_channels, inter_channels, L2=True)
        # Graph 16
        # self.gcn_3216 = CrossGraphConvolution(inter_channels, inter_channels//2, L2=True)
        # self.gcn_816 = CrossGraphConvolution(inter_channels, inter_channels//2, L2=True)
        self.gcn832_16 = CrossGraphSample(L2=True)
        # self.gcn_816 = CrossGraphSample(inter_channels, inter_channels, L2=True)
        # Graph 8
        # self.gcn_328 = CrossGraphConvolution(inter_channels, inter_channels//2, L2=True)
        # self.gcn_168 = CrossGraphConvolution(inter_channels, inter_channels//2, L2=True)
        self.gcn1632_8 = CrossGraphSample(L2=True)
        # self.gcn_168 = CrossGraphSample(inter_channels, inter_channels, L2=True)

        # Extend dim to in_channels, e.g.512
        self.extend_8 = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 1, padding=0, bias=False),
            BatchNorm2d(in_channels, eps=1e-4),
        )
        self.extend_16 = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 1, padding=0, bias=False),
            BatchNorm2d(in_channels, eps=1e-4),
        )
        self.extend_32 = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 1, padding=0, bias=False),
            BatchNorm2d(in_channels, eps=1e-4),
        )

        # Smooth Multi-scale feature
        self.smooth_s = nn.Sequential(
            nn.Conv2d(inter_channels*3, in_channels, 1, padding=0, bias=False),
            BatchNorm2d(in_channels),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
        )

        # Smooth Graph Feature
        self.smooth_g = nn.Sequential(
            nn.Conv2d(in_channels*4, in_channels, 3, padding=1, bias=False),
            BatchNorm2d(in_channels),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
        )

    def reproj(self, graph, Q, H, W):
        B, D, _ = graph.size()
        x = torch.matmul(graph, Q.permute(0, 2, 1)) # B, D, N
        x = x.reshape(B, D, H, W)
        return x

    def forward(self, c3, c4, c5):
        _, _, H, W = c3.size()

        # Projection
        graph_8, Q_8 = self.proj_8(c5)

        graph_16, Q_16 = self.proj_16(c4)

        graph_32, Q_32 = self.proj_32(c3)

        # GCN
        graph_8 = self.gcn_8(graph_8) * self.gamma8 + graph_8
        graph_16 = self.gcn_16(graph_16) * self.gamma16 + graph_16
        graph_32 = self.gcn_32(graph_32) * self.gamma32 + graph_32

        # Multi-scale in graph space
        # Graph 32
        graph_32_ = self.gcn816_32(torch.cat([graph_8, graph_16], dim=-1), graph_32)
        x_32_ = self.reproj(graph_32_, Q_32, H, W)
        # graph_1632 = self.gcn_1632(graph_16, graph_32)
        # Graph 16
        # graph_816 = self.gcn_816(graph_8, graph_16)
        graph_16_ = self.gcn832_16(torch.cat([graph_8, graph_32], dim=-1), graph_16)
        x_16_ = self.reproj(graph_16_, Q_16, H, W)
        # Graph 8
        # graph_168 = self.gcn_168(graph_16, graph_8)
        graph_8_ = self.gcn1632_8(torch.cat([graph_16, graph_32], dim=-1), graph_8)
        x_8_ = self.reproj(graph_8_, Q_8, H, W)

        # graph_32 = torch.cat([graph_32, graph_32_], dim=1)
        # graph_16 = torch.cat([graph_16, graph_16_], dim=1)
        # graph_8 = torch.cat([graph_8, graph_8_], dim=1)
        # graph_32 = torch.cat([graph_832, graph_1632], dim=1)
        # graph_16 = torch.cat([graph_816, graph_3216], dim=1)
        # graph_8 = torch.cat([graph_168, graph_328], dim=1)

        # Reprojection
        x_8 = self.reproj(graph_8, Q_8, H, W)
        x_8 = self.extend_8(x_8) + c5

        x_16 = self.reproj(graph_16, Q_16, H, W)
        x_16 = self.extend_16(x_16) + c4

        x_32 = self.reproj(graph_32, Q_32, H, W)
        x_32 = self.extend_32(x_32) + c3

        x_ms = self.smooth_s(torch.cat([x_8_, x_16_, x_32_], dim=1))

        # Smooth refinement
        output = self.smooth_g(torch.cat([x_8, x_16, x_32, x_ms], dim=1))

        return output

class CrossGraphSample(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    *** Cosine similarity adjcent matrix
    *** Cross different graphs
    """
    def __init__(self, L2=True):
        super(CrossGraphSample, self).__init__()
        # self.in_features = in_features
        # self.out_features = out_features
        self.L2 = L2
        # self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.bn = BatchNorm1d(256)
        nn.init.constant_(self.bn.weight, 0)
        nn.init.constant_(self.bn.bias, 0)

    #     if bias:
    #         self.bias = nn.Parameter(torch.FloatTensor(out_features))
    #     else:
    #         self.register_parameter('bias', None)
    #     self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.weight.size(1))
    #     self.weight.data.uniform_(-stdv, stdv)

    #     if self.bias is not None:
    #         self.bias.data.uniform_(-stdv, stdv)

    def norm_adj(self, input, target_g):
        if self.L2:
            adj_ = torch.matmul(F.normalize(target_g, p=2, dim=1).permute(0,2,1), F.normalize(input, p=2, dim=1))
        else:
            adj_ = torch.matmul(target_g.permute(0,2,1), input)
        adj_ = F.softmax(adj_, dim=-1) # -1   B, K_t, K_input
        # return adj_
        K = adj_.size(-1)
        k = int(round(K*0.8))
        _, idx = adj_.topk(dim=-1, k=k)
        adj = torch.zeros(adj_.size(), device=adj_.get_device())
        for b in range(adj.size(0)):
            for i in range(adj.size(1)):
                adj[b, i, idx[b, i]] = adj_[b, i, idx[b, i]]
        del adj_
        return adj

    def forward(self, input, target_g):
        adj = self.norm_adj(input, target_g) # K_target, K_input
        input = input.permute(0, 2, 1) # B, C, K_input -> B, K_input, C
        # support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, input).permute(0, 2, 1) # B, C, K_t
        # output = torch.matmul(output, self.weight).permute(0, 2, 1)
        return self.bn(self.relu(output)) * self.gamma + target_g
    #     if self.bias is not None:
    #         return self.bn(self.relu(output + self.bias)) 
    #     else:
    #         return self.bn(self.relu(output)) * self.gamma + target_g

    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' \
    #            + str(self.in_features) + ' -> ' \
    #            + str(self.out_features) + ')'

class CrossGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    *** Cosine similarity adjcent matrix
    *** Cross different graphs
    """
    def __init__(self, in_features, out_features, norm_layer=BatchNorm1d, bias=False, L2=True):
        super(CrossGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.L2 = L2
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)
        # self.conv = nn.Conv1d(in_features, out_features, 1, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)
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

    def norm_adj(self, input, target_g):
        if self.L2:
            adj = torch.matmul(F.normalize(target_g, p=2, dim=1).permute(0,2,1), F.normalize(input, p=2, dim=1))
        else:
            adj = torch.matmul(target_g.permute(0,2,1), input)
        adj = F.softmax(adj, dim=-1) # -1   B, K_t, K_input
        return adj

    def forward(self, input, target_g):
        adj = self.norm_adj(input, target_g)
        input = input.permute(0, 2, 1) # B, C, K_input -> B, K_input, C
        # support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, input)#.permute(0, 2, 1) # B, C, K_t
        # output = self.conv(output)
        output = torch.matmul(output, self.weight).permute(0, 2, 1)

        if self.bias is not None:
            return self.bn(self.relu(output + self.bias)) 
        else:
            return self.bn(self.relu(output))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
