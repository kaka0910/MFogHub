import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet
from torch.nn import functional as F
from .deeplabv3p import *
# from scse_original import SELayer, SCSEBlock

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel))
        self.spatial_se = nn.Conv2d(channel, 1, kernel_size=1,
                                    stride=1, padding=0, bias=False)

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        #print('bpool', x.size())
        chn_se = self.avg_pool(x).view(bahs, chs)
        #print('apool', chn_se.size())
        chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(bahs, chs, 1, 1))
        #print('x', x.size(), 'se', chn_se.size())
        chn_se = torch.mul(x, chn_se)
        #print('chn se', chn_se.size())
        #print('bspa_se', spa_se.size())
        spa_se = torch.sigmoid(self.spatial_se(x))
        #print('aspa_se', spa_se.size())
        spa_se = torch.mul(x, spa_se)
        #print('mspa_se', spa_se.size())
        #print('final_scse', torch.add(chn_se, 1, spa_se).size())
        return torch.add(chn_se, 1, spa_se)

class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x


class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionRefinementModule,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.SCSEBlockARM = SCSEBlock(128)

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        #x = self.SCSEBlockARM(x)
        return x

class LinkNet(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, criterion_seg, n_classes, n_channels):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(LinkNet, self).__init__()
        self.criterion_seg = criterion_seg
        self.nclass = n_classes

        # base = resnet.resnet18(pretrained=False)
        base = ResNet101(nInputChannels=3, os=16, pretrained=False)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        # self.encoder4 = base.layer4

        self.decoder1 = Decoder(256, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(512, 256, 3, 2, 1, 1)
        self.decoder3 = Decoder(1024, 512, 3, 2, 1, 1)
        # self.decoder4 = Decoder(2048, 1024, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.Conv2d(n_classes, n_classes, kernel_size=1, stride=1, padding=0)
        # self.lsm = nn.LogSoftmax(dim=1)
        # self.SCSEBlock4 = SCSEBlock(1024)
        self.SCSEBlock3 = SCSEBlock(512)
        self.SCSEBlock2 = SCSEBlock(256)
        self.SCSEBlock1 = SCSEBlock(64)
        self.ARM = AttentionRefinementModule(1024, 1024)


    def forward(self, x, y=None):
        # Initial block
        x = self.in_block(x)    # [2, 64, 256, 256]

        # Encoder blocks
        e1 = self.encoder1(x)    # [2, 256, 256, 256]
        e2 = self.encoder2(e1)   # [2, 512, 128, 128]
        e3 = self.encoder3(e2)   # [2, 1024, 64, 64]
        # e4 = self.encoder4(e3)   # [2, 2048, 64, 64]
        #e4 = self.ARM(e4)

        # Decoder blocks
        # d4 = self.decoder4(e4) + e3
        # d4 = self.SCSEBlock4(d4)
        # d3 = self.decoder3(d4) + e2
        d3 = self.decoder3(e3) + e2
        d3 = self.SCSEBlock3(d3)
        d2 = e1 + F.upsample(self.decoder2(d3), (e1.size(2), e1.size(3)), mode='bilinear')
        #d2 = self.decoder2(d3) + e1
        d2 = self.SCSEBlock2(d2)
        d1 = self.decoder1(d2) + x
        d1 = self.SCSEBlock1(d1)


        # Classifier
        out = self.tp_conv1(d1)
        out = self.conv2(out)
        out = self.tp_conv2(out)
        out = self.lsm(out)
        # import pdb; pdb.set_trace()

        if self.training:
            loss = self.criterion_seg(out, y.long())
            return out.max(1)[1].detach(), loss
        else:
            return out.detach()

        # return y

def get_scselink(dataset='h8',criterion_seg=None, **kwargs):
    # infer number of classes
    model = LinkNet(criterion_seg, n_classes=2, n_channels=3, **kwargs)
    return model