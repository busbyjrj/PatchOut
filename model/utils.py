# !/usr/bin/env python
# -*-coding:utf-8 -*-
# some codes reference from UTNet.


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.modules.utils import _pair
from torchinfo import summary


class LayerNorm(nn.Module):
    """
    From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_first", "channels_last"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(dim=1, keepdim=True)
            s = (x - u).pow(2).mean(dim=1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(inplanes),
                nn.ReLU(),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residue = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = out + self.shortcut(residue)

        return out


class BasicAttention(nn.Module):
    def __init__(
            self, in_channels, out_channels, image_size, heads=4, dim_head=64, dropout=0.0
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads

        self.ih, self.iw = image_size
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = depthwise_separable_conv(in_channels, in_channels * 3)

    def forward(self, x):
        B, C, H, W = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(
                t, "b (h d) ih iw -> b h (ih iw) d", h=self.heads, d=self.dim_head
            ),
            qkv,
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # use gather for more efficiency on GPUs
        # relative_bias = rearrange(self.relative_bias(), "(h w) c -> 1 c h w",
        #                           h=self.ih * self.iw, w=self.ih * self.iw)
        # dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h (ih iw) d -> b (h d) ih iw", iw=self.iw, ih=self.ih)

        return out


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=1):
        super().__init__()
        mlp_ratio = 1
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(
            dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio
        )
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.act(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class BasicTransformerBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, image_size, heads=4, dim_head=64, dropout=0.0
    ):
        super(BasicTransformerBlock, self).__init__()

        self.attn = BasicAttention(
            in_channels, out_channels, image_size, heads, dim_head, dropout
        )
        self.ff = MLP(out_channels, mlp_ratio=4)

        self.attn = PreNorm(out_channels, self.attn, LayerNorm)
        self.ff = PreNorm(out_channels, self.ff, LayerNorm)

    def forward(self, x):
        out = self.attn(x) + x
        out = self.ff(out) + out
        return out


class ReduceAttention(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            image_size,
            heads=4,
            dim_head=32,
            reduce_ratio=2,
            relative_pos=True,
    ):
        super().__init__()
        self.heads = heads
        # self.dim_head = dim_head
        self.dim_head = out_channels // heads
        self.inner_dim = self.dim_head * heads

        self.ih, self.iw = _pair(image_size)
        self.ih_reduce, self.iw_reduce = (
            self.ih // reduce_ratio,
            self.iw // reduce_ratio,
        )
        self.scale = self.dim_head ** -0.5

        self.relative_pos = relative_pos

        # if self.relative_pos:
        #     self.relative_bias = RelativePositionBias_v2(heads, (self.ih_reduce, self.iw_reduce))

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = depthwise_separable_conv(in_channels, self.inner_dim * 3)
        self.to_out = nn.Conv2d(self.inner_dim, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = rearrange(q, "b (h d) ih iw -> b h (ih iw) d", h=self.heads, d=self.dim_head)
        k, v = map(lambda t: F.interpolate(t, size=self.ih_reduce, mode="bilinear", align_corners=True), (k, v), )
        k, v = map(lambda t: rearrange(t, "b (h d) ih iw -> b h (ih iw) d", h=self.heads, d=self.dim_head), (k, v), )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # if self.relative_pos:
        #     relative_bias = self.relative_bias(H, W)
        #     dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h (ih iw) d -> b (h d) ih iw", iw=self.iw, ih=self.ih)

        out = self.to_out(out)
        return out


class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_ch,
            bias=bias,
            stride=stride,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out


class ReduceAttentionDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            image_size,
            heads=4,
            dim_head=32,
            reduce_ratio=1,
    ):
        super().__init__()
        self.heads = heads
        # self.dim_head = dim_head
        self.dim_head = out_channels // heads
        self.inner_dim = self.dim_head * heads

        self.ih, self.iw = _pair(image_size)
        self.ih_reduce, self.iw_reduce = (self.ih // reduce_ratio, self.iw // reduce_ratio,)
        self.scale = self.dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_q = depthwise_separable_conv(out_channels, self.inner_dim)
        self.to_kv = depthwise_separable_conv(in_channels, self.inner_dim * 2)

        self.to_out = nn.Conv2d(self.inner_dim, out_channels, kernel_size=1, bias=False)

    def forward(self, x, skip):
        # x: low-level feature map
        B, C, H, W = x.shape  # K, V
        # skip: high-level feature map
        BH, CH, HH, WH = skip.shape  # Q

        q = self.to_q(skip)
        q = rearrange(q, "b (h d) ih iw -> b h (ih iw) d", h=self.heads, d=self.dim_head)
        k, v = self.to_kv(x).chunk(2, dim=1)
        k, v = map(lambda t: F.interpolate(t, size=self.ih_reduce, mode="bilinear", align_corners=True), (k, v), )
        k, v = map(lambda t: rearrange(t, "b (h d) ih iw -> b h (ih iw) d", h=self.heads, d=self.dim_head), (k, v), )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h (ih iw) d -> b (h d) ih iw", iw=self.iw, ih=self.ih)

        out = self.to_out(out)
        return out


class ReduceTransformerBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            image_size,
            heads=4,
            dim_head=64,
            reduce_ratio=2,
            relative_pos=True,
    ):
        super(ReduceTransformerBlock, self).__init__()

        self.attn = ReduceAttention(
            in_channels,
            out_channels,
            image_size,
            heads,
            dim_head,
            reduce_ratio=reduce_ratio,
            relative_pos=relative_pos,
        )
        self.ff = MLP(out_channels, mlp_ratio=4)

        self.attn = PreNorm(out_channels, self.attn, LayerNorm)
        self.ff = PreNorm(out_channels, self.ff, LayerNorm)

    def forward(self, x):
        out = self.attn(x) + x
        out = self.ff(out) + out
        return out


class ReduceTransDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            image_size,
            heads=4,
            dim_head=64,
            reduce_ratio=1,
    ):
        super(ReduceTransDecoderBlock, self).__init__()
        # hidden_dim = int(in_channels * 4)

        self.bn_l = nn.BatchNorm2d(in_channels)
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.bn_mlp = nn.BatchNorm2d(out_channels)
        self.conv_ch = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.attn = ReduceAttentionDecoder(
            in_channels,
            out_channels,
            image_size,
            heads,
            dim_head,
            reduce_ratio=reduce_ratio,
        )
        self.ff = MLP(out_channels, mlp_ratio=4)

    def forward(self, x, skip):
        residue = F.interpolate(self.conv_ch(x), size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = self.bn_l(x)
        skip = self.bn_h(skip)
        out = self.attn(x, skip)
        out = out + residue
        residue = out
        out = self.bn_mlp(out)
        out = self.ff(out)
        out = out + residue
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth, downsample=False):
        super().__init__()

        self.layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                if downsample:
                    self.layers.append(nn.MaxPool2d(2))
                self.layers.append(BasicBlock(in_channels, out_channels))
            else:
                self.layers.append(BasicBlock(out_channels, out_channels))

    def forward(self, x):
        for block in self.layers:
            x = block(x)

        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2, upsample=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.ModuleList([])
        for i in range(depth - 1):
            self.layers.append(BasicBlock(out_channels, out_channels))

    def forward(self, x, skip):
        x_ = self.up(x)
        x_ = self.conv(x_)
        out = torch.cat([x_, skip], dim=1)
        out = self.conv2(out)
        for block in self.layers:
            out = block(out)
        return out


class TransBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            depth,
            heads,
            image_size,
            reduce_ratio,
            downsample=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        if downsample == True:
            self.layers.append(nn.MaxPool2d(2))
            self.layers.append(BasicBlock(in_channels, out_channels))
            image_size = (image_size[0] // 2, image_size[1] // 2)
        else:
            self.layers.append(BasicBlock(in_channels, out_channels, stride=1))
        for i in range(depth):
            self.layers.append(
                ReduceTransformerBlock(
                    out_channels,
                    out_channels,
                    heads=heads,
                    reduce_ratio=reduce_ratio,
                    image_size=image_size,
                )
            )

    def forward(self, x):
        for block in self.layers:
            x = block(x)

        return x


class UpTransBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            image_size,
            heads=4,
            dim_head=64,
            reduce_ratio=2,
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.trans = ReduceTransDecoderBlock(
            in_channels,
            out_channels,
            image_size=image_size,
            heads=heads,
            dim_head=dim_head,
            reduce_ratio=reduce_ratio,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.ModuleList([])
        for i in range(1 - 1):
            self.layers.append(BasicBlock(out_channels, out_channels))

    def forward(self, x, skip):
        out = self.trans(x, skip)
        out = torch.cat([out, skip], dim=1)
        out = self.conv2(out)
        for block in self.layers:
            out = block(out)
        return out
