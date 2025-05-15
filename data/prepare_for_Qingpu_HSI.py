# !/usr/bin/env python
# -*-coding:utf-8 -*-

import os
import numpy as np
from osgeo import gdal
from utils import create_image_by_roi, create_mask_for_testing_by_roi, sample_labels_all, create_testing_dataset

image_path = "./Qingpu_HSI_image.dat"
gt_path = "./Qingpu_HSI_gt.dat"
train_image_save_path = "./train/image"
train_gt_save_path = "./train/gt"
test_gt_save_path = "./test/Qingpu_HSI_test_gt.dat"
os.makedirs(train_image_save_path, exist_ok=True)
os.makedirs(train_gt_save_path, exist_ok=True)
os.makedirs(os.path.dirname(test_gt_save_path), exist_ok=True)

# read mean and std
lists = np.loadtxt("./Qingpu_HSI_mean_std.txt")
mean = lists[:, 0]
std = lists[:, 1]

# read rois
lists = np.loadtxt("./Qingpu_HSI_rois.txt")
rois_x = lists[:, 1]
rois_y = lists[:, 2]

# cut image by rois for training and validation
for index in range(len(rois_x)):
    img_save_name = os.path.join(
        train_image_save_path, "roi_image_" + str(index + 1) + ".dat")
    gt_save_name = os.path.join(
        train_gt_save_path, "roi_gt_" + str(index + 1) + ".dat")
    create_image_by_roi(rois_x[index], rois_y[index],
                        image_path, img_save_name, mean, std)
    create_image_by_roi(rois_x[index], rois_y[index], gt_path, gt_save_name)

# create testing gt by rois
create_mask_for_testing_by_roi(rois_x, rois_y, gt_path, test_gt_save_path)

# split training and validation in the cropped areas
train_samples = 5000
src_dir = r"./train/gt"
train_dir = r"./train/gt_train_" + str(train_samples)
val_dir = r"./train/gt_val_" + str(train_samples)
sample_labels_all(src_dir, train_dir, val_dir, train_samples)


# create testing dataset with padding
src_file = "./data/Qingpu_HSI_image.dat"
mean_std_file = "./data/Qingpu_HSI_mean_std.txt"
dst_file = "./data/test/Qingpu_HSI_image_norm_padding.dat"
create_testing_dataset(src_file, dst_file, mean_std_file)