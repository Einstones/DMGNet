#!/usr/bin/env python3
import torch
import torch.nn as nn

from base_module import conv_down, conv3x3, UNetConvBlock, SSA, Traditional_ssa
from pytorch_wavelets import DWTForward


class wave_ssa_net(nn.Module):

    def __init__(self, in_chn=3, wf=32, depth=5, relu_slope=0.2, mode='reflect'):
        super(wave_ssa_net, self).__init__()
        self.depth = depth
        self.down_path = []
        prev_channels = in_chn
        # The number 0 1 2 3 of each block are given by compute process
        self.down0 = UNetDownBlock(prev_channels, (2 ** 0) * wf, relu_slope)
        self.down1 = UNetDownBlock((2 ** 0) * wf, (2 ** 1) * wf, relu_slope)
        self.down2 = UNetDownBlock((2 ** 1) * wf, (2 ** 2) * wf, relu_slope)
        self.down3 = UNetDownBlock((2 ** 2) * wf, (2 ** 3) * wf, relu_slope)
        self.bottom = UNetConvBlock((2 ** 3) * wf, (2 ** 4) * wf, relu_slope)

        self.up0 = UNetUpBlock((2 ** 4) * wf, (2 ** 3) * wf, relu_slope)
        self.up1 = UNetUpBlock((2 ** 3) * wf, (2 ** 2) * wf, relu_slope)
        self.up2 = UNetUpBlock((2 ** 2) * wf, (2 ** 1) * wf, relu_slope)
        self.up3 = UNetUpBlock((2 ** 1) * wf, (2 ** 0) * wf, relu_slope)
        self.last = conv3x3((2 ** 0) * wf, prev_channels, bias=True)

        self.skip3 = nn.Sequential(UNetConvBlock((2 ** 0) * wf, (2 ** 0) * wf, relu_slope),
                                   UNetConvBlock((2 ** 0) * wf, (2 ** 0) * wf, relu_slope),
                                   UNetConvBlock((2 ** 0) * wf, (2 ** 0) * wf, relu_slope),
                                   UNetConvBlock((2 ** 0) * wf, (2 ** 0) * wf, relu_slope))
        self.skip2 = nn.Sequential(UNetConvBlock((2 ** 1) * wf, (2 ** 1) * wf, relu_slope),
                                   UNetConvBlock((2 ** 1) * wf, (2 ** 1) * wf, relu_slope),
                                   UNetConvBlock((2 ** 1) * wf, (2 ** 1) * wf, relu_slope))
        self.skip1 = nn.Sequential(UNetConvBlock((2 ** 2) * wf, (2 ** 2) * wf, relu_slope),
                                   UNetConvBlock((2 ** 2) * wf, (2 ** 2) * wf, relu_slope))
        self.skip0 = nn.Sequential(UNetConvBlock((2 ** 3) * wf, (2 ** 3) * wf, relu_slope))

        self.ssa0 = SSA((2 ** 3) * wf * 2)
        self.ssa1 = SSA((2 ** 2) * wf * 2)
        self.ssa2 = SSA((2 ** 1) * wf * 2)
        self.ssa3 = SSA((2 ** 0) * wf * 2)

    def forward(self, input_image):
        out0, out_down0 = self.down0(input_image)
        out1, out_down1 = self.down1(out_down0)
        out2, out_down2 = self.down2(out_down1)
        out3, out_down3 = self.down3(out_down2)
        out_bottom = self.bottom(out_down3)

        out_skip0 = self.skip0(out3)
        out_skip1 = self.skip1(out2)
        out_skip2 = self.skip2(out1)
        out_skip3 = self.skip3(out0)

        out_up0 = self.up0(out_bottom, out_skip0, self.ssa0)
        out_up1 = self.up1(out_up0, out_skip1, self.ssa1)
        out_up2 = self.up2(out_up1, out_skip2, self.ssa2)
        out_up3 = self.up3(out_up2, out_skip3, self.ssa3)
        output = self.last(out_up3)
        return output


class UNetDownBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetDownBlock, self).__init__()
        self.block = UNetConvBlock(in_size, out_size, relu_slope)
        self.down_sample = DWTForward(J=1, mode='reflect', wave='haar')

    def forward(self, x):
        out = self.block(x)
        out_down, _ = self.down_sample(out)
        return out, out_down


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, relu_slope)

    def forward(self, input_data, skip_data, ssa_block):
        up_data = self.up(input_data)
        ssa_data = ssa_block(up_data, skip_data)
        input_data = torch.cat([up_data, ssa_data], dim=1)
        out = self.conv_block(input_data)
        return out

    # 再把ssa替换为图像梯度图用于引导重建







