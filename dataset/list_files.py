# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : list_files.py
# Time       : 2022/8/4 20:09
# Author     : Renjie Ji
# Email      : busbyjrj@gmail.com
# Description:
# Reference  :
"""

import os

import numpy as np

# file_path = "../data/train/image"
# label_path = "../data/train/gt_train_5000"
file_path = "../data/train/image"
label_path = "../data/train/gt_val_5000"

file_list = os.listdir(label_path)
# with (open("./Qingpu_HSI_train.txt", "w") as f):
with (open("./Qingpu_HSI_val.txt", "w") as f):
    for line in file_list:
        if not (line.endswith(".npy")):
            continue
        label_name = line
        print(label_name)
        # image_name = line.replace("train_gt", "roi_image")
        image_name = line.replace("val_gt", "roi_image")

        data_path = os.path.join(file_path, image_name).replace("\\", "/")
        mask_path = os.path.join(label_path, label_name).replace("\\", "/")
        # 判断mask是否全部为0，如果是则不写入
        mask = np.load(mask_path)
        if np.sum(mask) == 0:
            continue
        # 判断image和mask是否都存在，如果不存在则不写入
        if not os.path.exists(data_path) or not os.path.exists(mask_path):
            print(data_path, mask_path, "not exist")
            continue
        f.writelines(image_name.split(".")[0] + " " + data_path + " " + mask_path + "\n")
        print(data_path, mask_path)
