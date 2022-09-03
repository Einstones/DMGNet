import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
import math
from scipy.special import comb
import numpy as np
import math
from torchvision.utils import make_grid
import cv2
from skimage import img_as_float32 as img_as_float
class ConvLayer1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer1, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, stride= stride)

        nn.init.xavier_normal_(self.conv2d.weight.data)

    def forward(self, x):
        return self.conv2d(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = (kernel_size-1)//2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
            nn.ReLU()
        )
        nn.init.xavier_normal_(self.block[0].weight.data)

    def forward(self, x):
        return self.block(x)

class transform_function(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(transform_function, self).__init__()
        self.ext = ConvLayer1(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1) # 保持大小不变
        self.pre = torch.nn.Sequential(
            ConvLayer1(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            ConvLayer1(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1),
        )

    def forward(self, x):
        y = self.ext(x)
        return y + self.pre(y)


class Inverse_transform_function(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Inverse_transform_function, self).__init__()
        self.ext = ConvLayer1(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1)
        self.pre = torch.nn.Sequential(
            ConvLayer1(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            ConvLayer1(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.pre(x)+x
        x = self.ext(x)
        return x

class SCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCA, self).__init__()
        self.conv_du = nn.Sequential(
                ConvLayer1(in_channels=channel, out_channels=channel // reduction, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                ConvLayer1(in_channels=channel // reduction, out_channels=channel, kernel_size=3, stride=1),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return y

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x

class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1, act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

class ConvProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout=0., last_stage=False, bias=True):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q, k, v


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


class LinearProjection_Concat_kv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        qkv_dec = self.to_qkv(x).reshape(B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv_enc = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k_d, v_d = qkv_dec[0], qkv_dec[1], qkv_dec[2]  # make torchscript happy (cannot use tensor as tuple)
        k_e, v_e = kv_enc[0], kv_enc[1]
        k = torch.cat((k_d, k_e), dim=2)
        v = torch.cat((v_d, v_e), dim=2)
        return q, k, v

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1), act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x = self.linear1(x)
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        x = self.dwconv(x)
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = self.linear2(x)
        return x

########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1), stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1), stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),  # 转置卷积
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=32, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

class MixOrderAttention(nn.Module):
    def __init__(self, k, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., se_layer=False):

        super().__init__()
        self.dim = dim
        self.k = k
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.se_layer = SELayer(dim) if se_layer else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)
        self.gating = nn.Linear(head_dim, self.k+1, bias=qkv_bias)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        # Order Gating Network
        gate = self.gating(x.mean(dim=1).reshape(B_, self.num_heads, C // self.num_heads))  # [759, 6, k]
        gate = self.sigmoid(gate).unsqueeze(-1).unsqueeze(-1) # [759, 6, k, 1, 1]
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-1)
        attn = (q @ (k.transpose(-2, -1)))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)
        attn = attn + torch.relu(relative_position_bias.unsqueeze(0))
        attn = (attn + attn.permute(0,1,3,2)) / 2
        d = torch.sum(attn, dim=3)
        d[d != 0] = torch.sqrt(1.0 / d[d != 0])
        attn = attn * d.unsqueeze(2) * d.unsqueeze(3)
        if self.k > 0:
            # Beinstein Approxiation
            E = torch.eye(attn.size(-1)).cuda()
            #L=I-D^(-0.5)AD^(-0.5)
            L = (E - attn)
            L_hat = (2 * E - L)
            tmp = []
            tmp.append(E)
            for i in range(self.k):
                E = E @ L_hat  # [E, (2E-L)^1, (2E-L)^2, ..., (2E-L)^k]
                tmp.append(E)
            out = ((comb(self.k, 0)/(2**self.k)) * gate[:, :, self.k, :, :]) * (tmp[self.k] @ v)
            E = torch.eye(attn.size(-1)).cuda()
            for j in range(self.k):
                E = E @ L
                out = out + ((comb(self.k, j+1)/(2**self.k)) * gate[:, :, j, :, :]) * ((E @ tmp[self.k-j-1]) @ v)
        else:
            out = attn @ v
        out = out.transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.se_layer(out)
        out = self.proj_drop(out)
        return out


## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=True):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class MORBlock(nn.Module):
    def __init__(self, k, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff',
                 se_layer=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = MixOrderAttention(k, dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection, se_layer=se_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * 4)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop) if token_mlp == 'ffn' else LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        #self.ffn = FeedForward(dim)


    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        attn_mask = None
        if self.shift_size > 0:
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, win_size*win_size, C

        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        #x = x.view(B, H * W, C)
        #x = x + self.drop_path(self.ffn(self.norm2(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #x = x.view(B, H * W, C)
        del attn_mask
        return x

class MORLayer(nn.Module):
    def __init__(self, k, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear', token_mlp='ffn', se_layer=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            MORBlock(k=k, dim=dim, input_resolution=input_resolution,
                                  num_heads=num_heads, win_size=win_size,
                                  shift_size=0 if (i % 2 == 0) else win_size // 2,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop, attn_drop=attn_drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                  norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,
                                  se_layer=se_layer)
            for i in range(depth)])

    def forward(self, x, mask=None):
        for blk in self.blocks:
            x = blk(x, mask)
        return x


class DMRNet(nn.Module):
    def __init__(self, k=1, img_size=128, in_chans=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='ffn', se_layer=True,
                 dowsample=Downsample, upsample=Upsample, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # Input/Output
        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=embed_dim, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = MORLayer(k=k, dim=embed_dim,
                                                output_dim=embed_dim,
                                                input_resolution=(img_size, img_size),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = MORLayer(k=k, dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size // 2, img_size // 2),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = MORLayer(k=k, dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // (2 ** 2), img_size // (2 ** 2)),
                                                depth=depths[2],
                                                num_heads=num_heads[2],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)
        self.encoderlayer_3 = MORLayer(k=k, dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 3), img_size // (2 ** 3)),
                                                depth=depths[3],
                                                num_heads=num_heads[3],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16)

        # Bottleneck
        self.conv = MORLayer(k=k, dim=embed_dim * 16,
                                      output_dim=embed_dim * 16,
                                      input_resolution=(img_size // (2 ** 4), img_size // (2 ** 4)),
                                      depth=depths[4],
                                      num_heads=num_heads[4],
                                      win_size=win_size,
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop_rate, attn_drop=attn_drop_rate,
                                      drop_path=conv_dpr,
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint,
                                      token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer)

        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = MORLayer(k=k, dim=embed_dim * 16,
                                                output_dim=embed_dim * 16,
                                                input_resolution=(img_size // (2 ** 3), img_size // (2 ** 3)),
                                                depth=depths[5],
                                                num_heads=num_heads[5],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[:depths[5]],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_1 = MORLayer(k=k, dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 2), img_size // (2 ** 2)),
                                                depth=depths[6],
                                                num_heads=num_heads[6],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_2 = MORLayer(k=k, dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // 2, img_size // 2),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.upsample_3 = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = MORLayer(k=k, dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size, img_size),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None):
        # Encoder
        y = self.input_proj(x)
        conv0 = self.encoderlayer_0(y, mask=mask)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0, mask=mask)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1, mask=mask)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2, mask=mask)
        pool3 = self.dowsample_3(conv3)

        # Bottleneck
        conv4 = self.conv(pool3, mask=mask)

        # Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0, mask=mask)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1, mask=mask)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2, mask=mask)

        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3, mask=mask)

        # Output Projection
        y = self.output_proj(deconv3)
        return x + y


class Gradient_Guidance(nn.Module):
    def __init__(self):
        super(Gradient_Guidance, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        # Fixed Conv
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)
        x = torch.cat(x_list, dim=1)
        return x

class GM(nn.Module):
    def __init__(self, channel):
        super(GM, self).__init__()
        self.cat = ConvLayer1(in_channels=channel*2, out_channels=channel, kernel_size=1, stride=1)
        self.C = ConvLayer1(in_channels=channel, out_channels=channel, kernel_size=3, stride=1)
        self.weight = SCA(channel)

    def forward(self, gra_prior, v):
        delta = self.weight(self.cat(torch.cat([self.C(gra_prior), v], 1)))
        return delta

class Fusion(nn.Module):
    def __init__(self, channel):
        super(Fusion, self).__init__()
        self.cat = ConvLayer1(in_channels=channel*2, out_channels=channel, kernel_size=1, stride=1)
        self.C1 = ConvLayer1(in_channels=channel, out_channels=channel, kernel_size=3, stride=1)
        self.C2 = ConvLayer1(in_channels=channel, out_channels=channel, kernel_size=3, stride=1)
        self.weight = SCA(channel)

    def forward(self, u, v):
        x = self.cat(torch.cat([self.C1(u), v], 1))
        result = self.weight(x) * x
        return result


class Prior(nn.Module):

    def __init__(self, channel=16):
        super(Prior, self).__init__()
        self.chanel_in = channel
        self.query_conv = ConvLayer1(in_channels=channel, out_channels=channel, kernel_size=3, stride=1)
        self.key_conv = ConvLayer1(in_channels=channel, out_channels=channel, kernel_size=3, stride=1)
        self.cat = ConvLayer1(in_channels=channel*2, out_channels=channel, kernel_size=3, stride=1)
        self.gamma1 = nn.Conv2d(channel * 2, 2, 3, 1, 1, bias=True)
        self.gamma2 = nn.Conv2d(channel * 2, 2, 3, 1, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, prior, x):        
        x_q = self.query_conv(x)
        prior_k = self.key_conv(prior)
        energy = x_q * prior_k
        # 代表x中包含的prior的程度
        attention = self.sig(energy)
        attention_x = x * attention
        attention_p = prior * attention
        x_gamma = self.gamma1(torch.cat((x, attention_x),dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]
        p_gamma = self.gamma2(torch.cat((prior, attention_p),dim=1))
        prior_out = prior * p_gamma[:, [0], :, :] + attention_p * p_gamma[:, [1], :, :]
        z = torch.cat((x_out, prior_out), dim=1)
        z = self.cat(z)
        return z

class DMGNet(nn.Module):
    def __init__(self, k=1, img_size=128, embed_dim=32, depths=None,
                 win_size=8, mlp_ratio=4., qkv_bias=True,
                 downsample=Downsample, upsample=Upsample):
        super(DMGNet, self).__init__()
        self.transform_function = transform_function(3, embed_dim)
        self.inverse_transform_function = Inverse_transform_function(embed_dim, 3)
        self.dmr = DMRNet(k=k, img_size=img_size, embed_dim=embed_dim, depths=depths,
                 win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                 token_projection='linear', token_mlp='leff',
                 downsample=downsample, upsample=upsample, se_layer=False)
        self.gm = nn.ModuleList([GM(embed_dim) for _ in range(2)])
        self.grad = Gradient_Guidance()
        self.fusion = Fusion(embed_dim)
        #self.prior = Prior(embed_dim)

    def forward(self, x, gt):
        x = self.transform_function(x)  # Frequent domain
        
        ####### Multiple Iterations #######
        for i in range(2):
            v = self.dmr(x)
            # Gradient and Feature Fusion
            # z = self.fusion(self.grad(x), x)
            # Gating Module
            z = self.fusion(self.grad(x), x)
            delta = self.gm[i](z, v)
            x = torch.mul((1 - 2 * delta), self.dmr(x)) + torch.mul(delta, z)
        
        x = self.inverse_transform_function(x)
        # Gradient Loss
        #pred_map = self.grad(self.inverse_transform_function(z)); gt_map = self.grad(gt)
        return x
