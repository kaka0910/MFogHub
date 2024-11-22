import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from ..nn import BatchNorm1d, BatchNorm2d

class RRPN(nn.Module):
    def __init__(self, in_channels=512, nodes=[8, 16, 32]):
        super(RRPN, self).__init__()
        inter_channels = 256 #256
        cat_channels = 256
        # Double Attention Proj - 5
        self.proj_8 = A2_Projection(in_channels, nodes[0])
        self.gcn_8 = GraphConvolution(inter_channels, inter_channels)
        self.connect_8 = nn.Sequential(
            nn.Conv2d(in_channels, cat_channels, 3, padding=1, bias=False),
            BatchNorm2d(cat_channels),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
        )

        # Double Attention Proj - 4
        # self.fuse4 = nn.Sequential(
        #     nn.Conv2d(in_channels+inter_channels, in_channels, 3, padding=1, bias=False),
        #     BatchNorm2d(in_channels),
        #     # nn.ReLU(inplace=True)
        #     nn.LeakyReLU(inplace=True)
        # )
        self.proj_16 = A2_Projection(in_channels, nodes[1])
        self.gcn_16 = GraphConvolution(inter_channels, inter_channels)
        self.connect_16 = nn.Sequential(
            nn.Conv2d(in_channels, cat_channels, 3, padding=1, bias=False),
            BatchNorm2d(cat_channels),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
        )

        # Double Attention Proj - 3
        # self.fuse3 = nn.Sequential(
        #     nn.Conv2d(in_channels+inter_channels, in_channels, 3, padding=1, bias=False),
        #     BatchNorm2d(in_channels),
        #     # nn.ReLU(inplace=True)
        #     nn.LeakyReLU(inplace=True)
        # )
        self.proj_32 = A2_Projection(in_channels, nodes[2])
        self.gcn_32 = GraphConvolution(inter_channels, inter_channels)
        self.connect_32 = nn.Sequential(
            nn.Conv2d(in_channels, cat_channels, 3, padding=1, bias=False),
            BatchNorm2d(cat_channels),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
        )

        # Channel attention
        self.se = SE((cat_channels + inter_channels)*3)
        # self.se = Spatial_SE((cat_channels + inter_channels)*3, reduction=8)

        self.smooth = nn.Sequential(
            nn.Conv2d((cat_channels + inter_channels)*3, 512, 3, padding=1, bias=False),
            BatchNorm2d(512),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
        )

    def reproj(self, graph, Q, H, W):
        B, D, _ = graph.size()
        x = torch.matmul(graph, Q.permute(0, 2, 1)) # B, D, N
        if H*W != x.size(-1):
            x = x.reshape(B, D, int(math.sqrt(x.size(-1))), int(math.sqrt(x.size(-1))))
            x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True)
        else:
            x = x.reshape(B, D, H, W)
        return x

    def forward(self, c3, c4, c5):
        _, _, H, W = c3.size()

        # Stage 5
        graph_8, Q_8 = self.proj_8(c5)
        graph_8 =self.gcn_8(graph_8)
        x_8 = self.reproj(graph_8, Q_8, H, W)
        c5 = self.connect_8(c5)

        # Stage 4
        # c4 = self.fuse4(torch.cat([c4, x_8], dim=1))
        c4 = torch.cat([c4, x_8], dim=1)
        graph_16, Q_16 = self.proj_16(c4)
        graph_16 = self.gcn_16(graph_16)
        x_16 = self.reproj(graph_16, Q_16, H, W)
        c4 = self.connect_16(c4)

        #Stage 3
        # c3 = self.fuse3(torch.cat([c3, x_16], dim=1))
        c3 = torch.cat([c3, x_16], dim=1)
        graph_32, Q_32 = self.proj_32(c3)
        graph_32 =self.gcn_32(graph_32)
        x_32 = self.reproj(graph_32, Q_32, H, W)
        c3 = self.connect_32(c3)

        output = torch.cat([x_8, c5, x_16, c4, x_32, c3], dim=1)
        output = self.se(output)
        output = self.smooth(output)

        return output

class RRPyramid(nn.Module):
    def __init__(self, in_channels=512, nodes=[8, 16, 32]):
        super(RRPyramid, self).__init__()
        inter_channels = 256 #256
        cat_channels = 256
        # Double Attention Proj - 5
        self.proj_8 = A2_Projection(in_channels, nodes[0])
        self.gcn_8 = GraphConvolution(inter_channels, inter_channels)
        self.connect_8 = nn.Sequential(
            nn.Conv2d(in_channels, cat_channels, 3, padding=1, bias=False),
            BatchNorm2d(cat_channels),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )

        # Double Attention Proj - 4
        self.proj_16 = A2_Projection(in_channels, nodes[1])
        self.gcn_16 = GraphConvolution(inter_channels, inter_channels)
        self.connect_16 = nn.Sequential(
            nn.Conv2d(in_channels, cat_channels, 3, padding=1, bias=False),
            BatchNorm2d(cat_channels),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )

        # Double Attention Proj - 3
        self.proj_32 = A2_Projection(in_channels, nodes[2])
        self.gcn_32 = GraphConvolution(inter_channels, inter_channels)
        self.connect_32 = nn.Sequential(
            nn.Conv2d(in_channels, cat_channels, 3, padding=1, bias=False),
            BatchNorm2d(cat_channels),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )

        self.smooth = nn.Sequential(
            nn.Conv2d((cat_channels + inter_channels)*3, 512, 3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )

    def reproj(self, graph, Q, H, W):
        B, D, _ = graph.size()
        x = torch.matmul(graph, Q.permute(0, 2, 1)) # B, D, N
        if H*W != x.size(-1):
            x = x.reshape(B, D, int(math.sqrt(x.size(-1))), int(math.sqrt(x.size(-1))))
            x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True)
        else:
            x = x.reshape(B, D, H, W)
        return x

    def forward(self, c3, c4, c5):
        _, _, H, W = c3.size()

        graph_32, Q_32 = self.proj_32(c3)
        graph_32 =self.gcn_32(graph_32)
        x_32 = self.reproj(graph_32, Q_32, H, W)
        c3 = self.connect_32(c3)

        graph_16, Q_16 = self.proj_16(c4)
        graph_16 = self.gcn_16(graph_16)
        x_16 = self.reproj(graph_16, Q_16, H, W)
        c4 = self.connect_16(c4)

        graph_8, Q_8 = self.proj_8(c5)
        graph_8 =self.gcn_8(graph_8)
        x_8 = self.reproj(graph_8, Q_8, H, W)
        c5 = self.connect_8(c5)

        output = torch.cat([x_8, c5, x_16, c4, x_32, c3], dim=1)
        output = self.smooth(output)

        return output

class GCU(nn.Module):
    def __init__(self, in_channels, nodes=[2, 4, 8, 16]):
        super(GCU, self).__init__()
        # Split Dynamic Code
        # self.proj_2 = Graph_split_projection(in_channels=in_channels, D=in_channels, K=nodes[0], num_down=0)
        self.proj_2 = A2_Projection(in_channels, nodes[0])
        self.gcn_2 = GraphConvolution(in_channels, in_channels)

        # Split Dynamic Code
        # self.proj_4 = Graph_split_projection(in_channels=in_channels, D=in_channels, K=nodes[1], num_down=0)
        self.proj_4 = A2_Projection(in_channels, nodes[1])
        self.gcn_4 = GraphConvolution(in_channels, in_channels)

        # Split Dynamic Code
        # self.proj_8 = Graph_split_projection(in_channels=in_channels, D=in_channels, K=nodes[2], num_down=0)
        self.proj_8 = A2_Projection(in_channels, nodes[2])
        self.gcn_8 = GraphConvolution(in_channels, in_channels)

        # Split Dynamic Code
        # self.proj_32 = Graph_split_projection(in_channels=in_channels, D=in_channels, K=nodes[3], num_down=0)
        self.proj_32 = A2_Projection(in_channels, nodes[3])
        self.gcn_32 = GraphConvolution(in_channels, in_channels)

        # Cross GCN k:32 -> 8
        self.cross328 = CrossGCN(k_key=nodes[2], k_query=nodes[3], in_channel=in_channels, gat=True)
        
        # Cross GCN k:8 -> 4
        self.cross84 = CrossGCN(k_key=nodes[1], k_query=nodes[2], in_channel=in_channels, gat=True)
        
        # Cross GCN k:4 -> 2
        self.cross42 = CrossGCN(k_key=nodes[0], k_query=nodes[1], in_channel=in_channels, gat=True)

        self.smooth = nn.Sequential(
            # nn.Dropout2d(0.1, False),
            nn.Conv2d(in_channels*5, 512, 3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def reproj(self, graph, Q, H, W):
        B, D, _ = graph.size()
        x = torch.matmul(graph, Q.permute(0, 2, 1)) # B, D, N
        if H*W != x.size(-1):
            x = x.reshape(B, D, int(math.sqrt(x.size(-1))), int(math.sqrt(x.size(-1))))
            x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True)
        else:
            x = x.reshape(B, D, H, W)
        return x

    def forward(self, x):
        """
            x: B, 256, H, W
            output: B, 256, H, W
        """
        _, _, H, W = x.size()

        graph_32, Q_32 = self.proj_32(x)
        graph_32 =self.gcn_32(graph_32)
        x_32 = self.reproj(graph_32, Q_32, H, W)

        graph_8, Q_8 = self.proj_8(x)
        graph_8 =self.gcn_8(graph_8)
        graph_8, Q_8 = self.cross328(graph_8, graph_32, Q_8, Q_32)
        x_8 = self.reproj(graph_8, Q_8, H, W)

        graph_4, Q_4 = self.proj_4(x)
        graph_4 = self.gcn_4(graph_4)
        graph_4, Q_4 = self.cross84(graph_4, graph_8, Q_4, Q_8)
        x_4 = self.reproj(graph_4, Q_4, H, W)

        graph_2, Q_2 = self.proj_2(x)
        graph_2 = self.gcn_2(graph_2)
        graph_2, Q_2 = self.cross42(graph_2, graph_4, Q_2, Q_4)
        x_2 = self.reproj(graph_2, Q_2, H, W)

        output = torch.cat([x_2, x_4, x_8, x_32, x], dim=1)
        output = self.smooth(output)

        return output

class Spatial_SE(nn.Module):
    def __init__(self, in_channels, reduction):
        super(Spatial_SE, self).__init__()
        self.fc = nn.Sequential(
            nn.AvgPool2d(4, 4),
            nn.Conv2d(in_channels, in_channels//reduction, 1, bias=False),
            # BatchNorm2d(in_channels//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False),
            # BatchNorm2d(in_channels)
        )
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        _, _, H, W = x.size()
        y = self.fc(x)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        y = self.sig(y)
        return x * y
        
class SE(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class A2_Projection(nn.Module):
    def __init__(self, in_channels, K):
        super(A2_Projection, self).__init__()
        self.in_channels = in_channels
        self.num_state = 256 #256
        self.K = K
        # self.phi = nn.Conv2d(self.in_channels, self.K, 1, bias=False)
        # self.theta = nn.Conv2d(self.in_channels, self.num_state, 1, bias=False)
        # self.rou = nn.Conv2d(self.in_channels, self.in_channels, 1, bias=False)
        self.phi = nn.Sequential(
            nn.Conv2d(self.in_channels, self.K, 1, bias=False),
            BatchNorm2d(self.K),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )
        self.theta = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_state, 1, bias=False),
            BatchNorm2d(self.num_state),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )
        self.rou = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_state, 1, bias=False),
            BatchNorm2d(self.num_state),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )
        self.value = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_state, 1, bias=False),
            BatchNorm2d(self.num_state),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=-1)
        # self.scale = nn.Parameter(torch.ones(1))
    
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
        x = self.value(x).reshape(B, self.num_state, H*W).unsqueeze(-1) # B, D, N, 1       #.expand(B, self.num_state, H*W, self.K).permute(1, 0, 2, 3) # C B N K
        Z = (((Q.unsqueeze(1) * x).sum(2))/Q.sum(1).unsqueeze(1)) #B, D, N, K -> B, D, K -> B, D, K /B, K -> B, D, K
        # Z = (((Q * x).sum(2))/Q.sum(1)).permute(1, 0, 2) #B, D, K /B, K = B, D, K
        
        # # TopK
        # x = (Q * x).topk(k=int(0.9*H*W), dim=2, sorted=False)[0]
        # Z = (x.sum(2) / Q.sum(1)) * self.scale
        # Z = Z.permute(1, 0, 2)

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

class Graph_split_projection(nn.Module):
    def __init__(self, in_channels, D=512, K=32, num_down=0):
        super(Graph_split_projection, self).__init__()
        self.D, self.K = D, K  # K is the number of codewords, D is the feature channel of each codeword
        self.in_channels = in_channels
        self.num_down = num_down

        # self.cam = SE(self.in_channels, 16)
        self.split_phi = AggGroup(self.in_channels, self.K, 8)
        self.conv_x = nn.Conv2d(self.in_channels, self.in_channels, 1, stride=1, padding=0, bias=False)
        self.theta = nn.Conv2d(self.in_channels, self.D, 1, bias=False)
        self.phi = nn.Conv2d(self.in_channels, self.K, 1, groups=self.K, bias=False) # , groups=self.K

        if self.num_down:
            down_ratio = 2 ** self.num_down
            down_size = 65 // down_ratio
            self.avg = nn.AdaptiveAvgPool2d(down_size)

        self.scale = nn.Parameter(torch.Tensor(D, K), requires_grad=True)
        self.scale.data.uniform_(-1, 0)

    def Gen_code(self, x):
        """
        x_phi: pixel -> node by conv_phi
        x_theta: p_channel -> node_channel by conv_theta
        """
        _, _, H, W = x.size()

        # x = self.cam(x)
        x_phi = self.split_phi(x)
        x_phi = self.phi(x).view(x.size(0), self.K, -1).permute(0, 2, 1).contiguous()
        x_theta = self.theta(x).view(x.size(0), self.D, -1).contiguous()
        x_phi = F.softmax(x_phi, dim=1)
        code = torch.matmul(x_theta, x_phi)#.div(H*W) # B, D, K
        code = F.normalize(code, p=2, dim=1) # eps=1e-8
        code = F.dropout(code, p=0.15)
        return code

    def embedded_residual(self, x):
        """
          x: B x D x H x W, N=HxW
          codes: B x D x K

          Z: B x D x K
          Q: B x N x K
          A: B, K, K
        """
        # Original version
        B, _, H, W = x.size()
        code_ = self.Gen_code(x)
        code = code_.unsqueeze(1).expand(B, H*W, self.D, self.K) # B, N, D, K
        x = x.reshape(x.size(0), x.size(1), H*W, 1).permute(0, 2, 1, 3).expand_as(code)
        residual = (x - code) * self.scale # B, N, D, K
        Q = F.softmax(-(residual.pow(2).sum(2)/2), 2) # B, N, K
        Z_ = (((Q * residual.permute(2, 0, 1, 3)).sum(2))/Q.sum(1)).permute(1, 0, 2) #B, D, K /B, K = B, D, K
        Z = (Z_.permute(1, 0, 2)/torch.sqrt((Z_.pow(2).sum(1)))).permute(1, 0, 2) #B, D, K

        # B, C, H, W = x.size()
        # code = self.Gen_code(x) # B, D, K
        # x = self.conv_x(x)
        # code = code.unsqueeze(1).expand(B, H*W, self.D, self.K) # B, N, D, K
        # x_ = x.reshape(x.size(0), x.size(1), H*W, 1).permute(0, 2, 1, 3).expand_as(code)
        # residual = x_ - code # * self.scale # B, N, D, K
        # residual = residual.pow(2).sum(2).div(C)
        # Q = F.softmax(-residual, 2) # B, N, K
        # x = x.reshape(B, C, H*W).unsqueeze(-1).expand(B, C, H*W, self.K).permute(1, 0, 2, 3)
        # Z = (((Q * x).sum(2))/Q.sum(1)).permute(1, 0, 2) #B, D, K /B, K = B, D, K
        # Z = F.normalize(Z, p=2, dim=1) #B, D, K

        return Z, Q

    def forward(self, X):
        """
            X: B, 256, H, W
        """
        assert X.dim() == 4
        # BxDxHxW
        if self.num_down:
            X = self.avg(X)
        # B x N x K
        Q, Z = self.embedded_residual(X)
        return Q, Z

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    *** Cosine similarity adjcent matrix
    """
    def __init__(self, in_features, out_features, norm_layer=BatchNorm1d, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)

        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(inplace=True)
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
        # adj = torch.matmul(F.normalize(input, p=2, dim=-1).permute(0,2,1), F.normalize(input, p=2, dim=-1))
        adj = torch.matmul(input.permute(0,2,1), input)
        adj = F.softmax(adj, dim=1) # -1
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
