from torch import nn
import torch
import torch.nn.functional as F

def get_Aspp_UNet(dataset='h8',criterion_seg=None, **kwargs):
    # infer number of classes
    model = Aspp_unet(criterion_seg, n_classes=2, in_channels=16, **kwargs)
    return model

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out

class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
 
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
 
        atrous_block6 = self.atrous_block6(x)
 
        atrous_block12 = self.atrous_block12(x)
 
        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

class Aspp_unet(nn.Module):
    def __init__(self, criterion_seg, n_classes, in_channels, deepsupervision=True, bilinear=True):
        super().__init__()

        self.criterion_seg = criterion_seg
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.deepsupervision = deepsupervision
        self.bilinear = bilinear
        self.nclass = n_classes

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.aspp = ASPP(nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(
            nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(
            nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(
            nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(
            nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(
            nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(
            nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(
            nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(
            nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(
            nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(
            nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(
                nb_filter[0], self.n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(
                nb_filter[0], self.n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(
                nb_filter[0], self.n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(
                nb_filter[0], self.n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)

    def forward(self, input, y=None):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0 = self.aspp(x4_0)## ASPP
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        
        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            outs = [output1, output2, output3, output4]

            if self.training:
                loss = 0
                for out in outs:
                    loss += self.criterion_seg(out, y.long())
                loss /= len(outs)
                return outs[-1].max(1)[1].detach(), loss
            else:
                return outs[-1].detach()
        
        else:
            out = self.final(x0_4)

            if self.training:
                loss = self.criterion_seg(out, y.long())
                return out.max(1)[1].detach(), loss
            else:
                return out.detach()