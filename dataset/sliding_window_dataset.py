# !/usr/bin/env python
# -*-coding:utf-8 -*-

import numpy as np
from torch.utils import data
from osgeo import gdal
import rasterio

class SlidingWindowDataset(data.Dataset):
    def __init__(self, image_path, grid, stride, preload=False):
        self.image_path = image_path
        self.grid = grid
        self.stride = stride
        self.preload = preload
        self.mean = np.loadtxt("./dataset/Qingpu_HSI_mean_std.txt")[:, 0]
        self.std = np.loadtxt("./dataset/Qingpu_HSI_mean_std.txt")[:, 1]

        with rasterio.open(self.image_path) as src:
            self.H, self.W = src.shape[:2]

        if self.preload:
            self.load_image()
            # self.pad_image()

        self.pos = self.create_sliding_window_pos()

    def load_image(self):
        ds = gdal.Open(self.image_path)
        self.image_MATRIX = ds.ReadAsArray()
        ds = None
        del ds
        # self.image_MATRIX = (self.image_MATRIX - self.mean[:, np.newaxis, np.newaxis]) / self.std[:, np.newaxis, np.newaxis]

    def __len__(self):
        return self.count_sliding_window()

    def __getitem__(self, index):
        if self.preload:
            i, j = self.pos[index]
            image = self.image_MATRIX[:, i:i + self.grid[0], j:j + self.grid[1]]
        else:
            with rasterio.open(self.image_path) as src:
                i, j = self.pos[index]
                image = src.read(window=((i, i + self.grid[0]), (j, j + self.grid[1])))
                image = (image - self.mean[:, np.newaxis, np.newaxis]) / self.std[:, np.newaxis, np.newaxis]
        image = image.astype(np.float32)
        return image, i, j

    def create_sliding_window_pos(self):
        grid_h, grid_w = self.grid
        stride_h, stride_w = self.stride
        pos = []
        for i in range(0, self.H - grid_h + 1, stride_h):
            for j in range(0, self.W - grid_w + 1, stride_w):
                pos.append((i, j))
        return pos

    def count_sliding_window(self):
        num_r = (self.H - self.grid[0]) // self.stride[0] + 1
        num_c = (self.W - self.grid[1]) // self.stride[1] + 1
        count = num_r * num_c
        return count

    def sliding_window(self):
        grid_h, grid_w = self.grid
        stride_h, stride_w = self.stride
        for i in range(0, self.H - grid_h + 1, stride_h):
            for j in range(0, self.W - grid_w + 1, stride_w):
                yield self.image_MATRIX[i:i + grid_h, j:j + grid_w], i, j