import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from einops.layers.torch import Reduce # Rearrange
# from .utils import pair

from encoding.utils.metrics import batch_pix_accuracy,batch_intersection_union

from functools import partial
import torch.nn.functional as F
nonlinearity = partial(F.relu, inplace=True)

from .mae_backbone import MaeBackbone
from .pointshift import *


def get_Mae_ori(dataset='cloud', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = MaeOri(img_size=256,in_chans=16,num_classes=datasets[dataset.lower()].NUM_CLASS,
        patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
         **kwargs)
    return model


class MaeOri(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False,criterion_seg=None,**kwargs):
        super(MaeOri, self).__init__(**kwargs)
        self.nclass=10
        self.global_pool = global_pool
        
        
        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']
        self.norm=nn.LayerNorm(embed_dim)
        self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                Conv1x1LN(3075, 512)    # 768 -> 512
            )
                # [xmq: dlink_vit_decoder]
        self.dblock = Dblock(512)
        
        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 10, 3, padding=1)

        # Criterion
        self.criterion_seg = criterion_seg
    def forward(self, x, y=None):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, -1, H//16, W//16)
        x=self.fpn4(x)
        e4 = self.dblock(x)
        # decoder
        d4 = self.decoder4(e4)
        d3 = self.decoder3(d4)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        if self.training:
            # out = out.float()
            loss = self.criterion_seg(out, y)
            return out.max(1)[1].detach(), loss
        else:
            return out.detach()

        return outcome
class Conv1x1LN(nn.Module):
    """
    1x1 Conv + LN 减少通道数

    ViTDet Page 15 - A.2 Implementation Details
    "Then for each pyramid level, we apply a 1x1 convolution with LN to reduce dimension
     to 256 and then a 3x3 convolution also with LN, similar to the per-level processing of FPN"
    """
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            Norm2d(out_channels)
        )
    
    def forward(self, x):
        return self.module(x)        
class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)   # NCHW -> NHWC
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
        return x

class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
        return out
class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
        return out

# [xmq: Dlink_vit DecoderBlock]
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x    