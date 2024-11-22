from torch import nn
import torch
import math
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=1024):
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
class Attention(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        dropout=0.1,
        block_size=1024
    ):
        super(Attention, self).__init__()
        self.n_head=8
        self.patch_embed=PatchEmbed()
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(block_size,
                                     block_size))

        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x, layer_past=None):
        x, (Hp, Wp)=self.patch_embed(x)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present   # TODO: check that this does not break anything
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
class AttnHead(nn.Module):
    def __init__(
        self,
        image_size=256,
        in_channels=16,
        embed_dim=32,
        norm_layer=nn.LayerNorm, 
        criterion_seg=None,
        backbone_type='base',
        pretrained_backbone=None,
        patch_size=16,
        dropout=0.1,
        block_size=64
    ):
        super(AttnHead, self).__init__()
        self.attn1=Attention()
        self.mlp = Mlp(in_features=1024, hidden_features=1024*4, act_layer=nn.GELU, drop=0)
        self.norm2 = nn.LayerNorm(1024)
    def forward(self,x):
        B,C,H,W=x.shape
        pres=None
        for i in range(C):
            
            out,pres=self.attn1(x[:,i:i+1],pres)
        out = out + self.mlp(self.norm2(out))
        return out
if __name__=='__main__':
    a=AttnHead()
    b=torch.rand((1,16,256,256))
    c=a(b)
    print(c.shape)