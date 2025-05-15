# !/usr/bin/env python
# -*-coding:utf-8 -*-


import torch
import torch.nn as nn
from thop import profile
from torchinfo import summary
from model.MSSSFF import MSSSFF
from model.utils import ConvBlock, TransBlock, UpTransBlock, UpConvBlock
from torch.nn.modules.utils import _pair


class PatchOut(nn.Module):
    def __init__(
            self,
            in_channels=251,
            base_channel=64,
            image_size=512,
            num_classes=1,
            reduce_ratio=2,
    ):
        super(PatchOut, self).__init__()

        # Encoder, we use CNN and Transformer blocks to build a hybrid encoder
        self.inc0 = ConvBlock(in_channels, base_channel, depth=2, downsample=False)
        self.down1 = ConvBlock(base_channel, base_channel * 2, depth=2, downsample=True)
        self.down2 = ConvBlock(base_channel * 2, base_channel * 4, depth=2, downsample=True)
        self.down3 = TransBlock(
            base_channel * 4,
            base_channel * 8,
            depth=2,
            heads=4,
            downsample=True,
            reduce_ratio=reduce_ratio * 2,
            image_size=(image_size // 4, image_size // 4),
        )
        self.down4 = TransBlock(
            base_channel * 8,
            base_channel * 8,
            depth=2,
            heads=4,
            downsample=True,
            reduce_ratio=reduce_ratio,
            image_size=(image_size // 8, image_size // 8),
        )

        # MSSSFF
        self.attn = MSSSFF(
            img_size=image_size,
            patch_size=[16, 8, 4, 2],
            channel_num=[64, 128, 256, 512],
            out_channels=128,
        )

        # Decoder
        self.up4 = UpConvBlock(in_channels=2 * base_channel, out_channels=base_channel, depth=1)
        self.up3 = UpConvBlock(in_channels=4 * base_channel, out_channels=2 * base_channel, depth=1)

        self.up2 = UpTransBlock(
            in_channels=8 * base_channel,
            out_channels=4 * base_channel,
            heads=2,
            image_size=_pair(image_size // 4),
            reduce_ratio=reduce_ratio * 2,
        )
        self.up1 = UpTransBlock(
            in_channels=8 * base_channel,
            out_channels=8 * base_channel,
            heads=2,
            image_size=_pair(image_size // 8),
            reduce_ratio=reduce_ratio,
        )

        self.outc = nn.Conv2d(base_channel, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc0(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1, x2, x3, x4, _ = self.attn(x1, x2, x3, x4)

        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)

        out = self.outc(out)
        return out


if __name__ == "__main__":
    img_rows, img_cols, img_nb = 512, 512, 251
    net = PatchOut(
        in_channels=img_nb,
        base_channel=64,
        num_classes=20,
        image_size=img_rows,
        reduce_ratio=4,
    )
    summary(net, input_size=(1, img_nb, img_rows, img_cols), depth=2)
    img = torch.randn(1, img_nb, img_rows, img_cols).cuda()
    flops, params = profile(net, inputs=(img,))
    print("param size = %f MB" % (params / 1000 / 1000.0))
    print("flops size = %f MB" % (flops / 1000 / 1000.0))
    print("-----------------------------------------------")
