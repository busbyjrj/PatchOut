# !/usr/bin/env python
# -*-coding:utf-8 -*-


import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import logging
import os

from osgeo import gdal
from tqdm import tqdm

from dataset.sliding_window_dataset import SlidingWindowDataset
from model.PatchOut import PatchOut
from utils.accuracy_analysis import accuracy_analysis
from utils.utils import load_pretrained_model

parser = argparse.ArgumentParser(description='train base net')
parser.add_argument('--net_name', default='PatchOut', help='network name')
parser.add_argument('--data_path', type=str, default='./datasets', help='path name of image dataset')
parser.add_argument('--dataset_name', type=str, default='Qingpu_HSI', help='dataset name')
parser.add_argument('--in_channels', type=int, default=251, help='input channels')
parser.add_argument('--image_size', type=int, default=512, help='image size')
parser.add_argument('--num_classes', type=int, default=20, help='number of classes')
parser.add_argument('--use_cuda', type=int, default=True, help='use cuda or not')
parser.add_argument('--save_path', type=str, default='./checkpoints', help='models and logs are saved here')

args, unparsed = parser.parse_known_args()

if torch.cuda.is_available() and args.use_cuda:
    use_cuda = True
else:
    use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

model = PatchOut(in_channels=args.in_channels, num_classes=args.num_classes,
                 image_size=args.image_size, reduce_ratio=4)
# model = torch.compile(model) # after torch 2.0

print('Loading the best model......')
checkpoint = torch.load(r"./checkpoints/model_best.pth.tar")
model = load_pretrained_model(model, checkpoint['net'])
model.to(device)
model.eval()

gt_path = r"./data/test/Qingpu_HSI_test_gt.dat"
image_path = r"./data/test/Qingpu_HSI_image_norm_padding.dat"

grid = (512, 512)
stride = (256, 256)
Dataset = SlidingWindowDataset(image_path, grid, stride, preload=True)

# test
start_time = time.time()
print('----------- Test --------------')
predictions = np.zeros((args.num_classes, Dataset.H, Dataset.W))
for i in tqdm(range(len(Dataset))):
    image, i, j = Dataset[i]
    image = torch.from_numpy(image).to(device)
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        output = output.cpu().numpy()
        output = output[0]
        predictions[:, i:i + grid[0], j:j + grid[1]] += output

src_ds = gdal.Open(gt_path)
cols = src_ds.RasterXSize
rows = src_ds.RasterYSize

# argmax
prediction = np.argmax(predictions, axis=0) + 1
# remove padding
start_x, start_y = 256, 256
end_x, end_y = cols + 256, rows + 256
class_matrix = prediction[start_y:end_y, start_x:end_x]

# save
save_path = r"./result/" + args.dataset_name + ".dat"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

dst_ds = gdal.GetDriverByName('ENVI').Create(save_path, cols, rows, 1, gdal.GDT_Int16)
dst_ds.GetRasterBand(1).WriteArray(class_matrix)
dst_ds.FlushCache()
dst_ds = None
src_ds = None

print("test time:" + str(time.time() - start_time))

cm, oa, e_acc, miou, fwIoU, kappa = accuracy_analysis(gt_path, save_path)
m_aa = np.mean(e_acc)

print("OA: " + str(oa))
print("AA: " + str(e_acc))
print("mAA: " + str(m_aa))
print("MIoU: " + str(miou))
print("FWIoU: " + str(fwIoU))
print("Kappa: " + str(kappa))
