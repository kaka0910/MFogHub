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
from clearml import Task

def get_Dlink_vit_v1(dataset='cloud',pretrained="", **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = Dlink_vit_v1(patch_size=16, num_classes=datasets[dataset.lower()].NUM_CLASS,pretrained=pretrained, **kwargs)
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
        patch_size=16,
        pretrained=""
    ):
        super(Dlink_vit_v1, self).__init__()
        self.base_size = 256
        self.crop_size = 256
        self.nclass = num_classes
        self.patch_size = patch_size

        # [xmq: dlink_vit_encoder]
        self.vit_backbone = ViTBackbone(
            img_size=256,
            patch_size=self.patch_size,
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
            pretrained=""
        )
        # '/root/share/Pretrain_model/Patch8/checkpoint-30.pth'
        # for name, param in self.vit_backbone.named_parameters():
        #     if name.startswith('pos_embed') or name.startswith('patch_embed') or name.startswith('blocks.0') or name.startswith('blocks.1') or name.startswith('blocks.2'):
        #         param.requires_grad = False
        
        # [xmq: dlink_vit_decoder]
        if self.patch_size==16:
            self.dblock = Dblock(512)

            filters = [64, 128, 256, 512]
        
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

        elif self.patch_size==8:
            self.dblock = Dblock(256)

            filters = [32, 64, 128, 256]
        
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
        if self.patch_size == 16:
            out = self.out_conv4(self.up4(center), e3)  # patch16:[16, 256, 16, 16]
            out = self.out_conv3(self.up3(out), e2)     # patch16:[16, 128, 32, 32]
            out = self.out_conv2(self.up2(out), e1)     # patch16:[16, 64, 64, 64]
            out = self.out_conv1(self.up1(out), self.up1(out))   # patch16:[16, 32, 128, 128]
            out = self.out_conv0(self.up0(out), self.up0(out))   # patch16:[16, 16, 256, 256]
            out = out.permute(0, 2, 3, 1).contiguous()
            out = self.norm_up(out)
            out = out.permute(0, 3, 1, 2).contiguous()
        elif self.patch_size == 8:
            out = self.out_conv4(self.up4(center), e3)  # patch8:[16, 128, 32, 32]
            out = self.out_conv3(self.up3(out), e2)     # patch8:[16, 64, 64, 64]
            out = self.out_conv2(self.up2(out), e1)     # patch8:[16, 32, 128, 128]
            out = self.out_conv0(self.up0(out), self.up0(out))   # patch8:[16, 16, 256, 256]
            out = out.permute(0, 2, 3, 1).contiguous()
            out = self.norm_up(out)
            out = out.permute(0, 3, 1, 2).contiguous()

        # head
        out = self.cls_head(out)
        # import pdb; pdb.set_trace()
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
    def __init__(self,channel,pretrained=None):
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
        if pretrained and pretrained.endswith('.pth'):
            self.load_pytorch_pretrained_dblock(pretrained)

    def load_pytorch_pretrained_dblock(self, pretrained_backbone_path):
        print('***In Dblock, Loading MAE pretrained backbone from', pretrained_backbone_path)
        pretrained_model = torch.load(pretrained_backbone_path, map_location='cuda')['model']
        state_dict = {}

        for key, value in pretrained_model.items():
            if key.startswith('dblock'):
                state_dict[key] = value

        self.load_state_dict(state_dict, strict=False)
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out # + dilate5_out
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
