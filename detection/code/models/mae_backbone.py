"""
https://github.com/ViTAE-Transformer/ViTDet/blob/main/mmdet/models/backbones/vit.py


modifications including:
    using LN for the convolutions in RPN and heads
    using large scale jittor for augmentation
    using RPE Relative Positional Embeddings) from MViT
    using longer training epochs and 1024 test size
    using global attention layers

There are other things to do:
    Implement the conv blocks for global information communication
    Tune the models for Cascade RCNN
    Train ViT models for the LVIS dataset
    Train ViTAE model with the ViTDet framework

MViT: RPE Relative Positional Embeddings)
ViTAE: 

"""

import math
import logging
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import drop_path, to_2tuple, trunc_normal_

_logger = logging.getLogger(__name__)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x,H,W, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x
class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.window_size = window_size
        q_size = window_size[0]
        kv_size = q_size
        rel_sp_dim = 2 * q_size - 1
        self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):      # H, W 没有被用到, 只是为了和 WindowAttention 保持一致
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def calc_rel_pos_spatial(
    attn,
    q,
    q_shape,
    k_shape,
    rel_pos_h,
    rel_pos_w,
    ):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        q_size = window_size[0]
        kv_size = window_size[1]
        rel_sp_dim = 2 * q_size - 1
        self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        x = x.reshape(B_, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size[0])  # nW*B, window_size, window_size, C
        x = x.view(-1, self.window_size[1] * self.window_size[0], C)  # nW*B, window_size*window_size, C
        B_w = x.shape[0]
        N_w = x.shape[1]
        qkv = self.qkv(x).reshape(B_w, N_w, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_w, N_w, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.view(-1, self.window_size[1], self.window_size[0], C)
        x = window_reverse(x, self.window_size[0], Hp, Wp)  # B H' W' C

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B_, H * W, C)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, window=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if not window:
            self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        else:
            self.attn = WindowAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None
        self.filter=GlobalFilter(dim, h=16, w=16)
    def forward(self, x, H, W):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=16, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)   # NCHW -> NHWC
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
        return x


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


class PropogationBasicBlock(nn.Module):
    """
    Cross-Window Propagation Block (Basic)

    ViTDet Page 6
    "Convolutional propagation. As an alternative, we add an extra convolutional
    block after each subset. A convolutional block is a residual block [26] that
    consists of one or more convolutions and an identity shortcut. The last layer in
    this block is initialized as zero, such that the initial status of the block is an
    identity [21]. Initializing a block as identity allows us to insert it into any place
    in a pre-trained backbone without breaking the initial status of the backbone."
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 将最后一层初始化为 0
        # self.conv2.weight.data.fill_(0)
        # self.conv2.bias.data.fill_(0)
        nn.init.constant_(self.conv2.weight, 0)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x, H, W):
        # 变换 shape
        batch_size, N, channel_cnt = x.shape
        x = x.reshape(batch_size, H, W, channel_cnt)
        x = x.permute((0, 3, 1, 2))    # NHWC -> NCHW

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        # 恢复原来的 shape
        out = out.permute((0, 2, 3, 1)) # NCHW -> NHWC
        out = out.reshape(batch_size, H*W, channel_cnt)

        return out


class MaeBackbone(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=16, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, init_values=None, use_checkpoint=False, 
                 use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 out_indices=[11], pretrained=None):
        super().__init__()
        self.num_classes = num_classes  # 暂无实际作用
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.out_indices = out_indices  # 暂无实际作用

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias    # 暂无实际作用
        self.use_checkpoint = use_checkpoint

        # block_interval = depth // 4     # 每个 subset 中的 block 数量
        # Global Propogation (perform global self-attention in the last block of each subset)
        # self.blocks = nn.ModuleList([
        #     Block(
        #         dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values,
        #         window_size=(14, 14) if ((i + 1) % block_interval != 0) else self.patch_embed.patch_shape,
        #         window=((i + 1) % block_interval != 0))
        #     for i in range(depth)])
        
        # 在所有 block 上使用 WindowAttention, 之后插入 Convolutional Propogation Block
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values,
                window_size=(14, 14), window=True)
            for i in range(depth)])

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self.norm = norm_layer(embed_dim)

        # 对应四个 scale: 
        if self.patch_size==16:
            dims = [64, 128, 256, 512]
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Conv1x1LN(embed_dim, dims[0])    # 768 -> 64
            )
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Conv1x1LN(embed_dim, dims[1])    # 768 -> 128
            )
            self.fpn3 = nn.Sequential(
                nn.Identity(),
                Conv1x1LN(embed_dim, dims[2])    # 768 -> 256
            )
            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                Conv1x1LN(embed_dim, dims[3])    # 768 -> 512
            )

        elif self.patch_size==8:
            dims = [32, 64, 128, 256]
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Conv1x1LN(embed_dim, dims[0])    # 768 -> 64
            )
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Conv1x1LN(embed_dim, dims[1])    # 768 -> 128
            )
            self.fpn3 = nn.Sequential(
                nn.Identity(),
                Conv1x1LN(embed_dim, dims[2])    # 768 -> 256
            )
            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                Conv1x1LN(embed_dim, dims[3])    # 768 -> 512
            )


        self.apply(self._init_weights)  # 对全连接层和LayerNorm层进行随机初始化
        self.fix_init_weight()

        if pretrained:
            if pretrained.endswith('.npz'):
                self.load_pretrained(pretrained)
            elif pretrained.endswith('.pth'):
                # import pdb; pdb.set_trace();
                self.load_pytorch_pretrained_backbone(pretrained)
        
        # 均匀地插入 4 个 Propagation Block (在加载预训练权重后再插入, 防止影响加载过程)
        block_interval = depth // 4
        for i in range(4):
            self.blocks.insert(block_interval * (i+1), PropogationBasicBlock(embed_dim, embed_dim))

    def fix_init_weight(self):
        """
        意义不明, 可能是某种 trick
        """
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        """
        对全连接层和LayerNorm层进行随机初始化
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def init_weights(self, pretrained=None):
    #     """Initialize the weights in backbone.

    #     Args:
    #         pretrained (str, optional): Path to pre-trained weights.
    #             Defaults to None.
    #     """
    #     pretrained = pretrained or self.pretrained
    #     if isinstance(pretrained, str):
    #         self.apply(self._init_weights)
    #         logger = get_root_logger()
    #         print(f"load from {pretrained}")
    #         load_checkpoint(self, pretrained, strict=False, logger=logger)
    #     elif pretrained is None:
    #         self.apply(self._init_weights)
    #     else:
    #         raise TypeError('pretrained must be a str or None')
    
    @torch.jit.ignore
    def load_pretrained(self, checkpoint_path, prefix=''):
        """加载预训练权重 ()
        """
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def load_pytorch_pretrained_backbone(self, pretrained_backbone_path):
        """Load PyTorch pretrained backbone
        """
        print('Loading MAE pretrained backbone from', pretrained_backbone_path)
        pretrained_model = torch.load(pretrained_backbone_path, map_location='cuda')['model']
        state_dict = {}

        # ################### interpolate pos_embed ###################
        pos_embed_checkpoint = pretrained_model['pos_embed']
        if pos_embed_checkpoint.shape[-2] % 2 == 1:
            pos_embed_checkpoint = pos_embed_checkpoint[:, 1:]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = self.patch_embed.num_patches
        num_extra_tokens = self.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            state_dict['pos_embed'] = new_pos_embed
        # ##############################################################

        for key, value in pretrained_model.items():
            if key.startswith('patch_embed') or key.startswith('norm') or key.startswith('blocks'):
                state_dict[key] = value

        self.load_state_dict(state_dict, strict=False)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)   # PatchEmbed 返回三个值: 结果、proj之后的height、proj之后的width
        # batch_size, seq_len, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, Hp, Wp)
            else:
                x = blk(x, Hp, Wp)

        x = self.norm(x)
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp)

        # 获取 4 个 scale 的 feature maps
        out=self.fpn4(xp)

        return out

    def forward(self, x):
        return self.forward_features(x)


# ##########################################################
# Borrow from timm.models.vision_transformer
# ##########################################################
def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


@torch.no_grad()
def _load_weights(model, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np
    from timm.models.helpers import adapt_input_conv

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    # 对于 patch_embed (patch embeddings):
    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))

    # ViTBackbone 中没有 cls_token
    # model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))

    # 对于 pos_embed (position embeddings):
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    

    model.pos_embed.copy_(pos_embed_w[:, :pos_embed_w.shape[1]-1])  # ViTBackbone 中没有 cls_token, 除去最后一个

    # 对于 Norm 层:
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))

    # ViTBackbone 中没有 Classifier head(s)
    # if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
    #     model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
    #     model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    
    # ViTBackbone 中没有 pre_logits
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    
    # 对于 transformer blocks:
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


if __name__ == '__main__':
    # 前向测试

    # Patch Size: 32
    # model = ViTBackbone(
    #     img_size=1024,
    #     patch_size=32,
    #     num_classes=2,
    #     embed_dim=768,
    #     depth=12,
    #     num_heads=12,
    #     mlp_ratio=4,
    #     qkv_bias=True,
    #     drop_rate=0.,
    #     attn_drop_rate=0.,
    #     drop_path_rate=0.1,
    #     use_abs_pos_emb=True,
    #     pretrained='/workspace/ViT_D-LinkNet/pretrained_weights/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'
    # )

    # Patch Size: 16
    # model = ViTBackbone(
    #     img_size=1024,
    #     patch_size=16,
    #     num_classes=2,
    #     embed_dim=768,
    #     depth=12,
    #     num_heads=12,
    #     mlp_ratio=4,
    #     qkv_bias=True,
    #     drop_rate=0.,
    #     attn_drop_rate=0.,
    #     drop_path_rate=0.1,
    #     use_abs_pos_emb=True,
    #     pretrained='/workspace/ViT_D-LinkNet/pretrained_weights/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'
    # )

    # ViT-Tiny
    model = ViTBackbone(
        img_size=256,
        patch_size=16,
        num_classes=10,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        use_abs_pos_emb=True,
        pretrained=None
    )

    x = torch.randn((2, 16, 256, 256))
    preds = model(x)

    # torch.Size([2, 768, 64, 64])
    print(preds[0].shape)   
    print(preds[1].shape)   
    print(preds[2].shape)   
    print(preds[3].shape)

    # torch.Size([2, 64, 256, 256])
    # torch.Size([2, 128, 128, 128])
    # torch.Size([2, 256, 64, 64])
    # torch.Size([2, 512, 32, 32])
