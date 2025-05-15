# !/usr/bin/env python
# -*-coding:utf-8 -*-

import os
import numpy as np
from osgeo import gdal

gt_dir = "./train/gt_train_5000"
gt_list = os.listdir(gt_dir)
gt_list = [os.path.join(gt_dir, gt_path) for gt_path in gt_list]

all_num_classes = {}
for gt_path in gt_list:
    if not (gt_path.endswith(".dat")):  # or gt_path.endswith(".npy")):
        continue
    if gt_path.endswith(".npy"):
        gt_MATRIX = np.load(gt_path)
    else:
        gt_MATRIX = gdal.Open(gt_path).ReadAsArray()

    # count the number of pixels of each class in each image
    num_classes = {}
    for i in np.unique(gt_MATRIX):
        n = len(gt_MATRIX[np.where(gt_MATRIX == i)])
        num_classes[i] = n
    print(gt_path, num_classes, gt_MATRIX.shape)

    # count the number of pixels of each class in all images
    for i in np.unique(gt_MATRIX):
        n = len(gt_MATRIX[np.where(gt_MATRIX == i)])
        if i in all_num_classes:
            all_num_classes[i] += n
        else:
            all_num_classes[i] = n

# sort num_classes
all_num_classes = {k: v for k, v in sorted(
    all_num_classes.items(), key=lambda item: item[0])}
print(all_num_classes)
