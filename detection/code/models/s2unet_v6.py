import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from einops.layers.torch import Reduce # Rearrange
# from .utils import pair

def get_s2unet_v6(dataset='cloud', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = S2UNet_v6(num_classes=datasets[dataset.lower()].NUM_CLASS, **kwargs)
    return model

class S2UNet_v6(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=4,
        in_channels=16,
        num_classes=10,
        embed_dim=64,
        norm_layer=nn.LayerNorm, 
        criterion_seg=None
    ):
        super(S2UNet_v6, self).__init__()
        self.base_size = 256
        self.crop_size = 256
        self.nclass = num_classes
        
        self.embed = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1)
        # self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim, norm_layer=norm_layer)
    
        self.down_4x = nn.Sequential(
            PatchMerging(dim=embed_dim, norm_layer=nn.LayerNorm),
            PatchMerging(dim=embed_dim*2, norm_layer=nn.LayerNorm)
        )
        self.proj = nn.Conv2d(embed_dim*4, embed_dim, kernel_size=1, stride=1)
        self.block = self.stage(dim_in=embed_dim, depth=4, expansion_factor=4, dropout = 0.5)
        self.down1 = self.down_stage(dim_in=embed_dim, depth=8, expansion_factor=4, dropout = 0.5)
        self.down2 = self.down_stage(dim_in=embed_dim*2, depth=27, expansion_factor=4, dropout = 0.5)
        self.down3 = self.down_stage(dim_in=embed_dim*4, depth=4, expansion_factor=4, dropout = 0.5)

        self.up3 = self.up_stage(dim_in=embed_dim*8, depth=4, expansion_factor=4, dropout = 0.5)
        self.up2 = self.up_stage(dim_in=embed_dim*4, depth=4, expansion_factor=4, dropout = 0.5)
        self.up1 = self.up_stage(dim_in=embed_dim*2, depth=4, expansion_factor=4, dropout = 0.5)

        # self.out_conv3 = self.out_conv(dim_in=embed_dim*8, dim_out=embed_dim*4, kernel_size=1, stride=1)
        # self.out_conv2 = self.out_conv(dim_in=embed_dim*4, dim_out=embed_dim*2, kernel_size=1, stride=1)
        # self.out_conv1 = self.out_conv(dim_in=embed_dim*2, dim_out=embed_dim, kernel_size=1, stride=1)

        self.out_conv3 = ConcatBackDim(dim=embed_dim*4)
        self.out_conv2 = ConcatBackDim(dim=embed_dim*2)
        self.out_conv1 = ConcatBackDim(dim=embed_dim)

        self.norm = nn.LayerNorm(embed_dim*8)
        self.norm_up = nn.LayerNorm(embed_dim)
        self.up_4x = FinalPatchExpand_X4(dim=embed_dim, dim_scale=4, norm_layer=nn.LayerNorm)

        self.cls_head = nn.Conv2d(embed_dim*2, num_classes, kernel_size=1, stride=1)

        # Criterion
        self.criterion_seg = criterion_seg

        # Initialize
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

    def stage(self, dim_in, depth=4, expansion_factor=3, dropout = 0.):
        return S2Block(dim_in, depth, expansion_factor, dropout = 0.)

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
        x0 = self.embed(x)
        x_ = self.down_4x(x0)
        x_ = self.proj(x_)

        # encoder
        x1 = self.block(x_)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # bottleneck
        center = self.down3(x3)
        center = center.permute(0, 2, 3, 1).contiguous()
        center = self.norm(center)
        center = center.permute(0, 3, 1, 2).contiguous()

        # decoder
        # out = self.out_conv3(torch.cat([self.up3(center), x3], dim=1))
        # out = self.out_conv2(torch.cat([self.up2(out), x2], dim=1))
        # out = self.out_conv1(torch.cat([self.up1(out), x1], dim=1))
        out = self.out_conv3(self.up3(center), x3)
        out = self.out_conv2(self.up2(out), x2)
        out = self.out_conv1(self.up1(out), x1)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = self.norm_up(out)
        out = out.permute(0, 3, 1, 2).contiguous()

        # up 4x
        out = self.up_4x(out)
        out = torch.cat([out, x0], dim=1)
        out = self.cls_head(out)

        if self.training:
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


class ConcatBackDim(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim*2, dim)

    def forward(self, x, y):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C)
        y = y.permute(0, 2, 3, 1).contiguous()
        y = y.view(B, -1, C)
        concat = torch.cat([x, y], dim=-1)
        out = self.linear(concat)
        out = out.view(B, H, W, C)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out


### S2-MLPv2
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def spatial_shift1(x):
    b,w,h,c = x.size()
    x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
    x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
    x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
    x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
    return x

def spatial_shift2(x):
    b,w,h,c = x.size()
    x[:,:,1:,:c//4] = x[:,:,:h-1,:c//4]
    x[:,:,:h-1,c//4:c//2] = x[:,:,1:,c//4:c//2]
    x[:,1:,:,c//2:c*3//4] = x[:,:w-1,:,c//2:c*3//4]
    x[:,:w-1,:,3*c//4:] = x[:,1:,:,3*c//4:]
    return x

class SplitAttention(nn.Module):
    def __init__(self, channel = 512, k = 3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias = False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias = False)
        self.softmax = nn.Softmax(1)
    
    def forward(self,x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)          #bs,k,n,c
        a = torch.sum(torch.sum(x_all, 1), 1)       #bs,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  #bs,kc
        hat_a = hat_a.reshape(b, self.k, c)         #bs,k,c
        bar_a = self.softmax(hat_a)                 #bs,k,c
        attention = bar_a.unsqueeze(-2)             # #bs,k,1,c
        out = attention * x_all                     # #bs,k,n,c
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out

class S2Attention(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention(channels)

    def forward(self, x):
        b, h, w, c = x.size()
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:,:,:,:c])
        x2 = spatial_shift2(x[:,:,:,c:c*2])
        x3 = x[:,:,:,c*2:]
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        return x

class S2Block(nn.Module):
    def __init__(self, d_model, depth, expansion_factor = 4, dropout = 0.):
        super().__init__()

        self.model = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(d_model, S2Attention(d_model)),
                PreNormResidual(d_model, nn.Sequential(
                    nn.Linear(d_model, d_model * expansion_factor),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * expansion_factor, d_model),
                    nn.Dropout(dropout)
                ))
            ) for _ in range(depth)]
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.model(x)
        x = x.permute(0, 3, 1, 2)
        return x


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
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=16, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        
        x = x.view(B, H//self.patch_size[0], W//self.patch_size[1], self.embed_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.permute(0, 2, 3, 1).contiguous() # B, H, W, C

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x) # B H/2*W/2 2*C

        x = x.view(B, H//2, W//2, 2 * self.dim)
        x = x.permute(0, 3, 1, 2) 

        return x

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H*W, C)

        x = self.expand(x) 

        x = x.view(B, H, W, 2*C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=2*C//4)
        x = x.view(B,-1,2*C//4)
        x = self.norm(x)
        
        x = x.view(B, H*2, W*2, C//2)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H*W, C)

        x = self.expand(x)

        x = x.view(B, H, W, 16*C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=16*C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        x = x.view(B, H*4, W*4, C)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

class FirstPatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.permute(0, 2, 3, 1).contiguous() # B, H, W, C

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x) # B H/2*W/2 2*C

        x = x.view(B, H//2, W//2, self.dim)
        x = x.permute(0, 3, 1, 2) 

        return x
