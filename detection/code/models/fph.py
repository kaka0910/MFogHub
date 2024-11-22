import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from ..nn import BatchNorm2d

class FPH(nn.Module):
    def __init__(self, in_channel=2048, out_channels=512, kernel_size=3, padding=1, dp=0.05):
        super(FPH, self).__init__()
        c5 = in_channel       # 2048
        c4 = in_channel // 2  # 1024
        c3 = in_channel // 4  # 512
        inter_channel = (c3 - out_channels) // 2

        # Head3 - Stage3
        self.head3_half = Head(c3, out_channels, kernel_size, padding, False, None)
        self.head3_4 = Head(c3, inter_channel, kernel_size, padding, False, None)
        self.head3_5 = Head(c3, inter_channel, kernel_size, padding, False, None)
        
        self.head3 = Head(c3, c3, kernel_size, padding, True, 0.05)
        
        # Head4 - Stage4
        self.head4_half = Head(c4, out_channels, kernel_size, padding, False, None)
        self.head4_3 = Head(c4, inter_channel, kernel_size, padding, False, None)
        self.head4_5 = Head(c4, inter_channel, kernel_size, padding, False, None)
        
        self.head4 = Head(c3, c3, kernel_size, padding, True, 0.05)
        
        # Head5 - Stage5
        self.head5_half = Head(c5, out_channels, kernel_size, padding, False, None)
        self.head5_3 = Head(c5, inter_channel, kernel_size, padding, False, None)
        self.head5_4 = Head(c5, inter_channel, kernel_size, padding, False, None)
        
        self.head5 = Head(c3, c3, kernel_size, padding, True, 0.05)

    def forward(self, c3, c4, c5):
        f3_half = self.head3_half(c3)
        f3_4 = self.head4_3(c4)
        f3_5 = self.head5_3(c5)

        f4_half = self.head4_half(c4)
        f4_3 = self.head3_4(c3)
        f4_5 = self.head5_4(c5)

        f5_half = self.head5_half(c5)
        f5_3 = self.head3_5(c3)
        f5_4 = self.head4_5(c4)

        f3 = self.head3(torch.cat([f3_half, f3_4, f3_5], dim=1))
        f4 = self.head4(torch.cat([f4_half, f4_3, f4_5], dim=1))
        f5 = self.head5(torch.cat([f5_half, f5_3, f5_4], dim=1))     
        return f3, f4, f5

class Head(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, use_drop, dp):
        super(Head, self).__init__()
        self.use_drop = use_drop
        if self.use_drop:
            self.dp = nn.Dropout2d(dp, False)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, bias=False),
            BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True)
        )
        # self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, bias=False)
        # self.bn = BatchNorm2d(out_channel)
        # self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        if self.use_drop:
            x = self.dp(x)
        # x = self.conv(x)
        # x = self.bn(x)
        # x = self.relu(x)
        x = self.layer(x)
        return x

