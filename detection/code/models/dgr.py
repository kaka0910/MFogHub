import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from mmcv.ops import ModulatedDeformConv2dPack, DeformConv2dPack


class CDGR(nn.Module):
    def __init__(self, in_channels=512, nodes=[8, 16, 32], norm_layer=nn.BatchNorm2d):
        super(CDGR, self).__init__()
        inter_channels = 256 #256
        cat_channels = 256
        # Double Attention Proj - 5
        self.proj_8 = A2_Projection(in_channels, nodes[0], norm_layer)

        # Double Attention Proj - 4
        self.proj_16 = A2_Projection(in_channels, nodes[1], norm_layer)

        # Double Attention Proj - 3
        self.proj_32 = A2_Projection(in_channels, nodes[2], norm_layer)

        # GCN / CGCN / GCN + CGCN
        self.gamma8 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.gamma16 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.gamma32 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.gcn_8 = GraphConvolution(inter_channels, inter_channels)
        self.gcn_16 = GraphConvolution(inter_channels, inter_channels)
        self.gcn_32 = GraphConvolution(inter_channels, inter_channels)
        # self.gcn_8 = GraphConvolutionTopK(inter_channels, inter_channels)
        # self.gcn_16 = GraphConvolutionTopK(inter_channels, inter_channels)
        # self.gcn_32 = GraphConvolutionTopK(inter_channels, inter_channels)

        # Extend dim to in_channels, e.g.512
        self.att_8 = Node_Attention(nodes[0])
        self.extend_8 = nn.Sequential(
            nn.Conv2d(cat_channels, in_channels, 1, padding=0, bias=False),
            norm_layer(in_channels),
        )
        self.att_16 = Node_Attention(nodes[1])
        self.extend_16 = nn.Sequential(
            nn.Conv2d(cat_channels, in_channels, 1, padding=0, bias=False),
            norm_layer(in_channels),
        )
        self.att_32 = Node_Attention(nodes[2])
        self.extend_32 = nn.Sequential(
            nn.Conv2d(cat_channels, in_channels, 1, padding=0, bias=False),
            norm_layer(in_channels),
        )

        # Smooth Graph Feature
        self.smooth_g = nn.Sequential(
            nn.Conv2d(in_channels*3, in_channels, 3, padding=1, bias=False),
            # ModulatedDeformConv2dPack(in_channels*3, in_channels, 3, padding=1, deformable_groups=1, bias=False),
            norm_layer(in_channels),
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
        _, _, H4, W4 = c4.size()
        _, _, H5, W5 = c5.size()

        # Projection
        graph_8, Q_8 = self.proj_8(c5)

        graph_16, Q_16 = self.proj_16(c4)

        graph_32, Q_32 = self.proj_32(c3)

        # GCN or CrossGCN or GCN+CrossGCN
        graph_8 = self.gcn_8(graph_8) * self.gamma8 + graph_8
        graph_16 = self.gcn_16(graph_16) * self.gamma16 + graph_16
        graph_32 = self.gcn_32(graph_32) * self.gamma32 + graph_32

        # Reprojection
        graph_8, att_8 = self.att_8(graph_8)
        x_8 = self.reproj(graph_8, Q_8, H5, W5)
        x_8 = self.extend_8(x_8) + c5

        graph_16, att_16 = self.att_16(graph_16)
        x_16 = self.reproj(graph_16, Q_16, H4, W4)
        x_16 = self.extend_16(x_16) + c4

        graph_32, att_32 = self.att_32(graph_32)
        x_32 = self.reproj(graph_32, Q_32, H, W)
        x_32 = self.extend_32(x_32) + c3

        # Smooth refinement
        output = self.smooth_g(torch.cat([x_8, x_16, x_32], dim=1))

        return output

class Node_Attention(nn.Module):
    def __init__(self, K):
        super(Node_Attention, self).__init__()
        self.K = K
        self.squeeze = nn.Conv1d(self.K, 1, kernel_size=1, bias=False)
        self.act = nn.PReLU(256)
        self.excite = nn.Sequential(
            nn.Conv1d(1, self.K, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        att = x.permute(0, 2, 1) # B, K, C
        att = self.squeeze(att).permute(0, 2, 1) # B, C, 1
        att = self.act(att).permute(0, 2, 1) # B, C, 1
        att = self.excite(att).permute(0, 2, 1) # B, C, K
        return x * att, att


class A2_Projection(nn.Module):
    def __init__(self, in_channels, K, norm_layer, num_state=256):
        super(A2_Projection, self).__init__()
        self.in_channels = in_channels
        self.K = K
        self.num_state = num_state #256

        self.phi = nn.Sequential(
            nn.Conv2d(self.in_channels, self.K, 1, bias=False),
            norm_layer(self.K),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
        )
        self.theta = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_state, 1, bias=False),
            norm_layer(self.num_state),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
        )
        self.rou = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_state, 1, bias=False),
            norm_layer(self.num_state),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
        )
        self.value = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_state, 1, bias=False),
            norm_layer(self.num_state),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=-1)

    def generate_graph(self, x):
        B, C, H, W = x.size()
        # Stage 1: generate discribtors
        x_phi = self.phi(x).view(B, -1, H*W) # B, K, N
        x_theta = self.theta(x).view(B, -1, H*W) # B, C, N
        x_theta = self.softmax(x_theta).permute(0, 2, 1) # B, N, C
        discrib = torch.matmul(x_phi, x_theta) # B, K, C
        # Stage2: Encode features
        x_rou = self.rou(x).view(B, -1, H*W) # B, C, N
        x_rou = self.softmax(x_rou.permute(0,2,1)).permute(0,2,1) # B, C, N
        Q = torch.matmul(discrib, x_rou).permute(0, 2, 1) # B, N, K
        # Stage3: extract similar regions by Q
        Q = F.normalize(Q, p=2, dim=1)
        Q = self.softmax(Q)
        x = self.value(x).reshape(B, self.num_state, H*W).unsqueeze(-1) # B, D, N, 1       
        Z = (((Q.unsqueeze(1) * x).sum(2))/Q.sum(1).unsqueeze(1)) #B, D, N, K -> B, D, K -> B, D, K /B, K -> B, D, K
        Z = F.normalize(Z, p=2, dim=1) #B, D, K
        # Z = F.dropout(Z, p=0.05)

        return Z, Q

    def forward(self, X):
        """
            X: B, 256, H, W
        """
        assert X.dim() == 4
        # BxDxHxW
        Z, Q = self.generate_graph(X)
        return Z, Q

class GraphConvolutionTopK(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    *** Cosine similarity adjcent matrix & TopK 0.8
    """
    def __init__(self, in_features, out_features, norm_layer=nn.BatchNorm1d, bias=False):
        super(GraphConvolutionTopK, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)
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

    def norm_adj(self, input):
        # adj_ = torch.matmul(F.normalize(input, p=2, dim=-1), F.normalize(input, p=2, dim=-1).permute(0,2,1))
        adj_ = torch.matmul(input, input.permute(0, 2, 1))
        adj_ = F.softmax(adj_, dim=-1) # -1   B, K, K
        K = adj_.size(1)
        k = int(round(K*0.9))
        _, idx = adj_.topk(dim=-1, k=k)
        adj = torch.zeros(adj_.size(), device=adj_.get_device())
        for b in range(adj.size(0)):
            for i in range(adj.size(1)):
                adj[b, i, idx[b, i]] = adj_[b, i, idx[b, i]]
        del adj_
        return adj

    def forward(self, input):
        input = input.permute(0, 2, 1) # B, C, K -> B, K, C
        adj = self.norm_adj(input)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support).permute(0, 2, 1)

        if self.bias is not None:
            return self.bn(self.relu(output + self.bias)) 
        else:
            return self.bn(self.relu(output))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    *** Cosine similarity adjcent matrix
    """
    def __init__(self, in_features, out_features, norm_layer=nn.BatchNorm1d, bias=False, L2=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.L2 = L2
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)

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

    def norm_adj(self, input):
        if self.L2:
            adj = torch.matmul(F.normalize(input, p=2, dim=-1), F.normalize(input, p=2, dim=-1).permute(0,2,1))
        else:
            adj = torch.matmul(input, input.permute(0,2,1))
        adj = F.softmax(adj, dim=-1) # -1   B, K, K
        return adj

    def forward(self, input):
        input = input.permute(0, 2, 1) # B, C, K -> B, K, C
        adj = self.norm_adj(input)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support).permute(0, 2, 1)

        if self.bias is not None:
            return self.bn(self.relu(output + self.bias)) 
        else:
            return self.bn(self.relu(output))

    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' \
    #            + str(self.in_features) + ' -> ' \
    #            + str(self.out_features) + ')'

class Upsample(nn.Module):
    def __init__(self, channels):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 1, bias=False)
    
    def forward(self, x):
        x = self.conv(x)
        return x