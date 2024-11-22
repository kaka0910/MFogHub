import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from einops.layers.torch import Reduce # Rearrange
# from .utils import pair

from encoding.utils.metrics import batch_pix_accuracy,batch_intersection_union

from functools import partial
import torch.nn.functional as F
nonlinearity = partial(F.relu, inplace=True)

from .vit_backbone import ViTBackbone
from .pointshift import *


def get_Dlink_vit_v1(dataset='cloud', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = Dlink_vit_v1(num_classes=datasets[dataset.lower()].NUM_CLASS, **kwargs)
    return model


class Dlink_vit_v1(nn.Module):
    params = {
        'tiny':  {'embed_dim': 192, 'depth': 12,'num_heads': 3},
        'small': {'embed_dim': 384, 'depth': 12,'num_heads': 6},
        'base':  {'embed_dim': 768, 'depth': 12,'num_heads': 12},
        'large': {'embed_dim': 1024,'depth': 24,'num_heads': 16}
    }
    def __init__(
        self,
        image_size=256,
        in_channels=16,
        num_classes=10,
        embed_dim=32,
        norm_layer=nn.LayerNorm, 
        criterion_seg=None,
        backbone_type='base',
        pretrained_backbone=None,
    ):
        super(Dlink_vit_v1, self).__init__()
        self.base_size = 256
        self.crop_size = 256
        self.nclass = num_classes
        
        # [xmq: dlink_vit_encoder]
        self.vit_backbone = ViTBackbone(
            img_size=256,
            patch_size=16,
            num_classes=10,
            embed_dim=self.params[backbone_type]['embed_dim'],
            depth=self.params[backbone_type]['depth'],
            num_heads=self.params[backbone_type]['num_heads'],
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            use_abs_pos_emb=True,
            pretrained='/root/share/Pretrain_model/checkpoint-201.pth'
        )
        
        # [xmq: dlink_vit_decoder]
        self.dblock = Dblock(512)
        
        filters = [64, 128, 256, 512]
        # self.decoder4 = DecoderBlock(filters[3], filters[2])
        # self.decoder3 = DecoderBlock(filters[2], filters[1])
        # self.decoder2 = DecoderBlock(filters[1], filters[0])
        # self.decoder1 = DecoderBlock(filters[0], filters[0])
        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        # self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        # self.finalconv3 = nn.Conv2d(32, 10, 3, padding=1)
        
        self.up4 = self.up_stage(dim_in=filters[3], depth=2, expansion_factor=4, dropout = 0.12)
        self.up3 = self.up_stage(dim_in=filters[2], depth=2, expansion_factor=4, dropout = 0.08)
        self.up2 = self.up_stage(dim_in=filters[1], depth=2, expansion_factor=4, dropout = 0.04)
        self.up1 = self.up_stage(dim_in=filters[0], depth=2, expansion_factor=4, dropout = 0.)
        self.up0 = self.up_stage(dim_in=32, depth=2, expansion_factor=4, dropout = 0.)
        
        self.out_conv4 = ConcatBackDim(dim=filters[2])
        self.out_conv3 = ConcatBackDim(dim=filters[1])
        self.out_conv2 = ConcatBackDim(dim=filters[0])
        self.out_conv1 = ConcatBackDim(dim=32)
        self.out_conv0 = ConcatBackDim(dim=16)
        
        self.norm = nn.LayerNorm(filters[3])
        self.norm_up = nn.LayerNorm(16)
    
        self.cls_head = nn.Conv2d(16, num_classes, kernel_size=1, stride=1)

        # Criterion
        self.criterion_seg = criterion_seg

        # [xmq:pointshift_Initialize]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def down_stage(self, dim_in, depth=2, expansion_factor=3, dropout = 0.):
        return nn.Sequential(
            PatchMerging(dim=dim_in, norm_layer=nn.LayerNorm),
            S2Block(dim_in*2, depth, expansion_factor, dropout = 0.)
        )

    def up_stage(self, dim_in, depth=2, expansion_factor=3, dropout = 0.):
        return nn.Sequential(
            PatchExpand(dim=dim_in, dim_scale=2, norm_layer=nn.LayerNorm),
            S2Block(dim_in//2, depth, expansion_factor, dropout = 0.)
        )
    
    def out_conv(self, dim_in, dim_out, kernel_size=1, stride=1):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x, y=None):
        # encoder
        e1, e2, e3, e4 = self.vit_backbone(x)

        # Center
        center = self.dblock(e4)
        center = center.permute(0, 2, 3, 1).contiguous()
        center = self.norm(center)
        center = center.permute(0, 3, 1, 2).contiguous()

        # import pdb; pdb.set_trace()
        out = self.out_conv4(self.up4(center), e3)
        out = self.out_conv3(self.up3(out), e2)
        out = self.out_conv2(self.up2(out), e1)
        out = self.out_conv1(self.up1(out), self.up1(out))
        out = self.out_conv0(self.up0(out), self.up0(out))
        out = out.permute(0, 2, 3, 1).contiguous()
        out = self.norm_up(out)
        out = out.permute(0, 3, 1, 2).contiguous()

        # head
        out = self.cls_head(out)

        if self.training:
            # out = out.float()
            loss = self.criterion_seg(out, y)
            return out.max(1)[1].detach(), loss
        else:
            return out.detach()
    
    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union
        
    def load_mae_pretrained(self, pretrained_weight_path):
        print('Loading MAE pretrained encoder and decoder from', pretrained_weight_path)
        pretrained_model = torch.load(pretrained_weight_path, map_location='cpu')['model']
        state_dict = {}

        for key, value in pretrained_model.items():
            # Position Embedding
            if key == 'pos_embed':
                # num_patches + 1 -> num_patches
                state_dict['vit_backbone.pos_embed'] = value[:, :value.shape[1]-1, :]
                
            if key.startswith('patch_embed') or key.startswith('norm') or key.startswith('fpn'):
                state_dict['vit_backbone.' + key] = value
            
            # Transformer blocks in ViT backbone (except for propogation blocks)
            elif key.startswith('blocks.'):
                block_interval = 12 // 4    # depth=12
                conv_prop_block_idx = [block_interval * (i+1) for i in range(4)]
                block_idx = int(key.split('.')[1])
                if block_idx not in conv_prop_block_idx:
                    state_dict['vit_backbone.' + key] = value
                else:
                    print('Droppig: ', key)
            
            elif key.startswith('dblock') or key.startswith('decoder4') or \
                key.startswith('decoder3') or key.startswith('decoder2') or \
                key.startswith('decoder1') or key.startswith('finaldeconv1') or \
                key.startswith('finalconv2'):
                state_dict[key] = value
            
            else:
                print('Droppig: ', key)

        # The final convolution layer does not need to load
        # state_dict['finalconv3.weight'] = self.finalconv3.weight
        # state_dict['finalconv3.bias'] = self.finalconv3.bias

        self.load_state_dict(state_dict, strict=False)


# [xmq: Dlink_vit Dblock]
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

# class ConcatBackDim(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#         self.linear = nn.Linear(dim*2, dim)

#     def forward(self, x, y):
#         B, C, H, W = x.shape
#         x = x.permute(0, 2, 3, 1).contiguous()
#         x = x.view(B, -1, C)
#         y = y.permute(0, 2, 3, 1).contiguous()
#         y = y.view(B, -1, C)
#         concat = torch.cat([x, y], dim=-1)
#         out = self.linear(concat)
#         out = out.view(B, H, W, C)
#         out = out.permute(0, 3, 1, 2).contiguous()
#         return out


# ### S2-MLPv2
# class PreNormResidual(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x):
#         return self.fn(self.norm(x)) + x

# def spatial_shift1(x):
#     b,w,h,c = x.size()
#     x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
#     x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
#     x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
#     x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
#     return x

# def spatial_shift2(x):
#     b,w,h,c = x.size()
#     x[:,:,1:,:c//4] = x[:,:,:h-1,:c//4]
#     x[:,:,:h-1,c//4:c//2] = x[:,:,1:,c//4:c//2]
#     x[:,1:,:,c//2:c*3//4] = x[:,:w-1,:,c//2:c*3//4]
#     x[:,:w-1,:,3*c//4:] = x[:,1:,:,3*c//4:]
#     return x

# class SplitAttention(nn.Module):
#     def __init__(self, channel = 512, k = 3):
#         super().__init__()
#         self.channel = channel
#         self.k = k
#         self.mlp1 = nn.Linear(channel, channel, bias = False)
#         self.gelu = nn.GELU()
#         self.mlp2 = nn.Linear(channel, channel * k, bias = False)
#         self.softmax = nn.Softmax(1)
    
#     def forward(self,x_all):
#         b, k, h, w, c = x_all.shape
#         x_all = x_all.reshape(b, k, -1, c)          #bs,k,n,c
#         a = torch.sum(torch.sum(x_all, 1), 1)       #bs,c
#         hat_a = self.mlp2(self.gelu(self.mlp1(a)))  #bs,kc
#         hat_a = hat_a.reshape(b, self.k, c)         #bs,k,c
#         bar_a = self.softmax(hat_a)                 #bs,k,c
#         attention = bar_a.unsqueeze(-2)             # #bs,k,1,c
#         out = attention * x_all                     # #bs,k,n,c
#         out = torch.sum(out, 1).reshape(b, h, w, c)
#         return out

# class S2Attention(nn.Module):
#     def __init__(self, channels=512):
#         super().__init__()
#         self.mlp1 = nn.Linear(channels, channels * 3)
#         self.mlp2 = nn.Linear(channels, channels)
#         self.split_attention = SplitAttention(channels)

#     def forward(self, x):
#         b, h, w, c = x.size()
#         x = self.mlp1(x)
#         # x1 = spatial_shift1(x[:,:,:,:c])
#         x1 = x[:,:,:,:c]
#         # x2 = spatial_shift2(x[:,:,:,c:c*2])
#         x2 = x[:,:,:,c:c*2]
#         # x3 = x[:,:,:,c*2:]
#         x3 = x[:,:,:,c*2:]
#         x_all = torch.stack([x1, x2, x3], 1)
#         a = self.split_attention(x_all)
#         x = self.mlp2(a)
#         return x

# class S2Block(nn.Module):
#     def __init__(self, d_model, depth, expansion_factor = 4, dropout = 0.):
#         super().__init__()

#         self.model = nn.Sequential(
#             *[nn.Sequential(
#                 PreNormResidual(d_model, S2Attention(d_model)),
#                 PreNormResidual(d_model, nn.Sequential(
#                     nn.Linear(d_model, d_model * expansion_factor),
#                     nn.GELU(),
#                     nn.Dropout(dropout),
#                     nn.Linear(d_model * expansion_factor, d_model),
#                     nn.Dropout(dropout)
#                 ))
#             ) for _ in range(depth)]
#         )

#     def forward(self, x):
#         x = x.permute(0, 2, 3, 1)
#         x = self.model(x)
#         x = x.permute(0, 3, 1, 2)
#         return x


'''
class S2MLPv2(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=[7, 2],
        in_channels=3,
        num_classes=1000,
        d_model=[192, 384],
        depth=[4, 14],
        expansion_factor = [3, 3],
    ):
        image_size = pair(image_size)
        oldps = [1, 1]
        for ps in patch_size:
            ps = pair(ps)
            assert (image_size[0] % (ps[0] * oldps[0])) == 0, 'image must be divisible by patch size'
            assert (image_size[1] % (ps[1] * oldps[1])) == 0, 'image must be divisible by patch size'
            oldps[0] = oldps[0] * ps[0]
            oldps[1] = oldps[1] * ps[1]
        assert (len(patch_size) == len(depth) == len(d_model) == len(expansion_factor)), 'patch_size/depth/d_model/expansion_factor must be a list'
        super().__init__()

        self.stage = len(patch_size)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else d_model[i - 1], d_model[i], kernel_size=patch_size[i], stride=patch_size[i]),
                S2Block(d_model[i], depth[i], expansion_factor[i], dropout = 0.)
            ) for i in range(self.stage)]
        )

        self.mlp_head = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(d_model[-1], num_classes)
        )

    def forward(self, x):
        embedding = self.stages(x)
        out = self.mlp_head(embedding)
        return out
'''

#######
# class PatchEmbed(nn.Module):
#     r""" Image to Patch Embedding
#     Args:
#         img_size (int): Image size.  Default: 224.
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """

#     def __init__(self, img_size=224, patch_size=4, in_chans=16, embed_dim=96, norm_layer=nn.LayerNorm):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.patches_resolution = patches_resolution
#         self.num_patches = patches_resolution[0] * patches_resolution[1]

#         self.in_chans = in_chans
#         self.embed_dim = embed_dim

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         if norm_layer is not None:
#             self.norm = norm_layer(embed_dim)
#         else:
#             self.norm = None

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # FIXME look at relaxing size constraints
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
#         if self.norm is not None:
#             x = self.norm(x)
        
#         x = x.view(B, H//self.patch_size[0], W//self.patch_size[1], self.embed_dim)
#         x = x.permute(0, 3, 1, 2).contiguous()
#         return x

#     def flops(self):
#         Ho, Wo = self.patches_resolution
#         flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
#         if self.norm is not None:
#             flops += Ho * Wo * self.embed_dim
#         return flops

# class PatchMerging(nn.Module):
#     r""" Patch Merging Layer.
#     Args:
#         dim (int): Number of input channels.
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """

#     def __init__(self, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
#         self.norm = norm_layer(4 * dim)

#     def forward(self, x):
#         """
#         x: B, C, H, W
#         """
#         B, C, H, W = x.shape
#         assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

#         x = x.permute(0, 2, 3, 1).contiguous() # B, H, W, C

#         x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
#         x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
#         x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
#         x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
#         x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
#         x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

#         x = self.norm(x)
#         x = self.reduction(x) # B H/2*W/2 2*C

#         x = x.view(B, H//2, W//2, 2 * self.dim)
#         x = x.permute(0, 3, 1, 2) 

#         return x

# class PatchExpand(nn.Module):
#     def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
#         self.norm = norm_layer(dim // dim_scale)

#     def forward(self, x):
#         """
#         x: B, C, H, W
#         """
#         B, C, H, W = x.shape
#         x = x.permute(0, 2, 3, 1).contiguous()
#         x = x.view(B, H*W, C)

#         x = self.expand(x) 

#         x = x.view(B, H, W, 2*C)
#         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=2*C//4)
#         x = x.view(B,-1,2*C//4)
#         x = self.norm(x)
        
#         x = x.view(B, H*2, W*2, C//2)
#         x = x.permute(0, 3, 1, 2).contiguous()

#         return x

# class FinalPatchExpand_X4(nn.Module):
#     def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.dim_scale = dim_scale
#         self.expand = nn.Linear(dim, 16*dim, bias=False)
#         self.output_dim = dim 
#         self.norm = norm_layer(self.output_dim)

#     def forward(self, x):
#         """
#         x: B, C, H, W
#         """
#         B, C, H, W = x.shape
#         x = x.permute(0, 2, 3, 1).contiguous()
#         x = x.view(B, H*W, C)

#         x = self.expand(x)

#         x = x.view(B, H, W, 16*C)
#         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=16*C//(self.dim_scale**2))
#         x = x.view(B,-1,self.output_dim)
#         x= self.norm(x)

#         x = x.view(B, H*4, W*4, C)
#         x = x.permute(0, 3, 1, 2).contiguous()

#         return x

# class FirstPatchMerging(nn.Module):
#     r""" Patch Merging Layer.
#     Args:
#         dim (int): Number of input channels.
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """

#     def __init__(self, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.reduction = nn.Linear(4 * dim, dim, bias=False)
#         self.norm = norm_layer(4 * dim)

#     def forward(self, x):
#         """
#         x: B, C, H, W
#         """
#         B, C, H, W = x.shape
#         assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

#         x = x.permute(0, 2, 3, 1).contiguous() # B, H, W, C

#         x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
#         x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
#         x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
#         x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
#         x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
#         x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

#         x = self.norm(x)
#         x = self.reduction(x) # B H/2*W/2 2*C

#         x = x.view(B, H//2, W//2, self.dim)
#         x = x.permute(0, 3, 1, 2) 

#         return x
