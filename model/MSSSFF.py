# !/usr/bin/env python
# -*-coding:utf-8 -*-
# some codes reference from UCTransNet.


import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from model.utils import ReduceTransformerBlock


class Patch_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(
            self, patch_size, img_size, in_channels, out_channels, dropout_rate=0.1
    ):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        # DWConv instead
        self.patch_embeddings = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=patch_size,
                stride=patch_size,
                groups=in_channels,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
        )

        # 固定的位置编码
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))

        return x


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0

        # DWConv instead
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=in_channels,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=in_channels,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
        )

        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        self.weight = nn.Parameter(torch.ones(2))

    def forward(self, x_C, x):
        x_C = nn.Upsample(scale_factor=self.scale_factor)(x_C)
        weight = F.softmax(self.weight, dim=0)
        x_C = self.conv(x_C)
        out = weight[0] * x_C + weight[1] * x
        out = self.norm(out)
        out = self.activation(out)
        return out


class MSSSFF(nn.Module):
    def __init__(
            self,
            img_size=256,
            patch_size=[16, 8, 4, 2],
            channel_num=[64, 128, 256, 512],
            out_channels=64,
    ):
        super().__init__()
        self.patchSize_1 = patch_size[0]
        self.patchSize_2 = patch_size[1]
        self.patchSize_3 = patch_size[2]
        self.patchSize_4 = patch_size[3]
        self.embeddings_1 = Patch_Embeddings(
            self.patchSize_1,
            img_size=img_size,
            in_channels=channel_num[0],
            out_channels=out_channels,
        )
        self.embeddings_2 = Patch_Embeddings(
            self.patchSize_2,
            img_size=img_size // 2,
            in_channels=channel_num[1],
            out_channels=out_channels,
        )
        self.embeddings_3 = Patch_Embeddings(
            self.patchSize_3,
            img_size=img_size // 4,
            in_channels=channel_num[2],
            out_channels=out_channels,
        )
        self.embeddings_4 = Patch_Embeddings(
            self.patchSize_4,
            img_size=img_size // 8,
            in_channels=channel_num[3],
            out_channels=out_channels,
        )

        self.reconstruct_1 = Reconstruct(
            out_channels, channel_num[0], kernel_size=3, scale_factor=patch_size[0]
        )
        self.reconstruct_2 = Reconstruct(
            out_channels, channel_num[1], kernel_size=3, scale_factor=patch_size[1]
        )
        self.reconstruct_3 = Reconstruct(
            out_channels, channel_num[2], kernel_size=3, scale_factor=patch_size[2]
        )
        self.reconstruct_4 = Reconstruct(
            out_channels, channel_num[3], kernel_size=3, scale_factor=patch_size[3]
        )

        self.C_layers = nn.ModuleList()

        for i in range(4):
            self.C_layers.append(
                ReduceTransformerBlock(
                    4 * out_channels,
                    4 * out_channels,
                    heads=4,
                    reduce_ratio=1,
                    image_size=img_size // patch_size[0],
                )
            )

    def forward(self, x1, x2, x3, x4):
        x1_ = self.embeddings_1(x1)
        x2_ = self.embeddings_2(x2)
        x3_ = self.embeddings_3(x3)
        x4_ = self.embeddings_4(x4)
        x_all_C = torch.cat([x1_, x2_, x3_, x4_], dim=1)

        for layer in self.C_layers:
            x_all_C = layer(x_all_C)

        x1_C, x2_C, x3_C, x4_C = torch.split(
            x_all_C, [x1_.size(1), x2_.size(1), x3_.size(1), x4_.size(1)], dim=1
        )

        x1 = self.reconstruct_1(x1_C, x1)
        x2 = self.reconstruct_2(x2_C, x2)
        x3 = self.reconstruct_3(x3_C, x3)
        x4 = self.reconstruct_4(x4_C, x4)

        return x1, x2, x3, x4, None
