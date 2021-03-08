"""
unet.py - Model and module class for unet and unet++ model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.upsampling import Upsample

class DoubleConv(nn.modules):
    def __init__(self, ch_in, ch_out, up_flag = False):
        super.__init__()
        ch_mid = ch_in // 2 if up_flag else ch_in * 2
        self.double_conv = nn.Sequential(
            nn.conv2d(ch_in, ch_mid, 3),
            nn.BatchNorm2d(ch_mid // 2),
            nn.ReLU(inplace=True),
            nn.conv2d(ch_mid, ch_out, 3),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x

class DownSample(nn.modules):
    def __init__(self, ch_in, ch_out):
        super.__init()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(ch_in, ch_out)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x

class UpSample(nn.modules):
    def __init__(self, ch_in, ch_out, bilinear = True):
        super.__init__()
        if bilinear:
            self.up_conv = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
            self.conv = DoubleConv(ch_in, ch_out // 2, up_flag = True)
        else:
            self.up_conv = nn.ConvTranspose2d(ch_in, ch_in // 2, kernel_size = 2, stride = 2)
            self.conv = DoubleConv(ch_in, ch_out, up_flag = True)
    
    def forward(self, x1, x2):
        x1 = self.up_conv(x1)

        deltax = x2.size()[3] - x1.size()[3]
        deltay = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [deltax // 2, deltax - deltax // 2,
                        deltay // 2, deltay - deltay // 2])
        
        x = torch.cat([x2, x1], dim = 1)
        x = self.conv(x)

        return x

class OutConv(nn.module):
    def __init__(self, ch_in, num_classes):
        super.__init__()
        self.out_conv = nn.Conv2d(ch_in, num_classes, kernel_size = 1, stride = 1)
    
    def forward(self, x):
        x = self.out_conv(x)
        return x


class Unet(nn.modules):

    def __init__(self, ch_in, num_classes, up_mode = "bilinear"):
        super(Unet, self).__init__()
        self.in = DoubleConv(ch_in, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        factor = 2 if up_mode == "bilinear" else 1
        self.down4 = DownSample(512, 1024 // factor)
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)
        self.out = OutConv(64, num_classes)

    def forward(self, x):
        x0 = self.in(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        x7 = self.up3(x6, x1)
        x8 = self.up4(x7, x0)
        out_map = self.out(x8)

        return out_map

