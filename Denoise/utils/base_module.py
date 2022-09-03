import torch
import torch.nn as nn
import torch.functional as F
# from pytorch_wavelets import DWTForward
import os


class ConvLayer(nn.Module):
    """
    convolution module
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        """

        """
        super(ConvLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        nn.init.xavier_normal_(self.conv2d.weight.data)

    def forward(self, x):
        return self.conv2d(x)


def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


def dilationConv(in_chn, out_chn, dilation, bias=True):
    # padding =
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, dilation=dilation, padding=dilation, bias=bias)
    return layer


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc
        return out


class DilationConvBlock(nn.Module):

    def __init__(self, in_size, out_size, dilation, relu_slope):
        super(DilationConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, dilation=dilation, padding=dilation, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(out_size, out_size, kernel_size=3, dilation=dilation, padding=dilation, bias=True),
            nn.LeakyReLU(relu_slope))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc
        return out


class SSA(nn.Module):

    def __init__(self, in_size, hidden_channel=16, relu_slope=0.2):
        super(SSA, self).__init__()
        self.blocks = UNetConvBlock(in_size, hidden_channel, relu_slope)
        self.hidden_channel = hidden_channel

    def forward(self, up_data, skip_data):
        ssa_data = torch.cat([up_data, skip_data], dim=1)
        ssa_data = self.blocks(ssa_data)

        b_, c_, h_, w_ = skip_data.shape
        V_t = ssa_data.reshape(b_, self.hidden_channel, h_ * w_)
        V_t = V_t / (1e-6 + torch.abs(V_t).sum(dim=2, keepdim=True))
        V = V_t.transpose(2, 1)
        mat = torch.matmul(V_t, V)
        mat_inv = torch.inverse(mat)
        project_mat = torch.matmul(mat_inv, V_t)
        skip_data = skip_data.reshape(b_, c_, h_ * w_)
        project_feature = torch.matmul(project_mat, skip_data.transpose(2, 1))

        skip_data = torch.matmul(V, project_feature).transpose(2, 1).reshape(b_, c_, h_, w_)
        return skip_data


# class Traditional_ssa(nn.Module):
#
#     def __init__(self, mode):
#         super(Traditional_ssa, self).__init__()
#         self.wave_transformer = DWTForward(J=1, mode=mode, wave='haar')
#
#     def forward(self, up_data, skip_data):
#         tssa_data = torch.cat([up_data, skip_data], dim=1)
#         tssa_data = self.wave_transformer(tssa_data)
#         return tssa_data


class ResidualBlock(nn.Module):
    # reference to Residual Dense Network for Image Restoration
    # residual and dense block
    def __init__(self, in_channel, out_channel, relu_slope):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
        )
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=True)

    def forward(self):
        return self


class DenseBlock(nn.Module):
    # reference to Residual Dense Network for Image Restoration
    # residual and dense block
    def __init__(self, in_channel, out_channel, relu_slope):
        super(DenseBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
        )
        self.conv = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=True)

    def forward(self):
        return self


def adaptive(model, pretrain_path, noisy_image, criterion, optimizer, iteration):
    pretrain_path = os.path.join(pretrain_path, 'best.pth')
    model.load_state_dict(torch.load(pretrain_path))
    for i in range(iteration):
        prediction = model(noisy_image)
        model.zero_grad()
        optimizer.zero_grad()
        loss = criterion(prediction, noisy_image)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = model(noisy_image)
        prediction = torch.clamp(prediction, 0., 1.)
    return prediction



# 0 增加noise estimation模块

# 2 增加传统模块。如小波变换等，进行低频图片抽取 haar小波 维纳滤波, gamma校正, some ISP function
# 3 增加random sample from N(0, 1)模块，可采用noisy image estimation结果约束选择参数
# 4 设计卷积方式使得特征抽取以及噪声抽取均不通过上采样以及下采样的操作，同时产生一个问题即获取的low resolution images大小是否会发生变化呢
# 接上， 是不是可以在小波变换得到low resolution图之后进行简单的super resolution操作将图变大到跟特征图一致呢
# 5 通过图像梯度约束 3
# 6 RDN from Residual Dense Network for Image Restoration -- 增加residual dense block
# 7 noise flow:
# 8 k-gamma解耦照片与ISO


