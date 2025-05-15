# !/usr/bin/env python
# -*-coding:utf-8 -*-

from sklearn import metrics
from osgeo import gdal
import numpy as np
from sklearn.metrics import confusion_matrix


def accuracy_analysis(gt_path, res_path):
    gt_MATRIX = gdal.Open(gt_path).ReadAsArray().reshape(-1).astype(np.int16)
    res_MATRIX = gdal.Open(res_path).ReadAsArray().reshape(-1).astype(np.int16)

    num_classes = np.max(gt_MATRIX)

    # remote 0
    ignored_labels = [0]
    ignored_mask = np.zeros(gt_MATRIX.shape[:2], dtype=bool)
    for l in ignored_labels:
        ignored_mask[gt_MATRIX == l] = True
    ignored_mask = ~ignored_mask

    gt_MATRIX = gt_MATRIX[ignored_mask]
    res_MATRIX = res_MATRIX[ignored_mask]


    cm = confusion_matrix(gt_MATRIX.flatten(),
                          res_MATRIX.flatten(),
                          labels=range(1, num_classes + 1))

    # Compute per-class accuracy
    e_acc = [cm[x][x] / np.sum(cm[x, :]) for x in range(len(cm))]

    # Compute global accuracy
    total = np.sum(cm)
    oa = sum([cm[x][x] for x in range(len(cm))])
    oa *= 100 / float(total)

    # miou
    iou = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    miou = np.nanmean(iou)

    # FWIoU
    freq = np.sum(cm, axis=1) / np.sum(cm)
    fwIoU = (freq[freq > 0] * iou[freq > 0]).sum()

    # Kappa
    kappa = metrics.cohen_kappa_score(gt_MATRIX, res_MATRIX)

    return cm, oa, e_acc, miou, fwIoU, kappa


if __name__ == "__main__":
    gt_path = r"./data/test/Qingpu_HSI_test_gt.dat"
    res_path = r"./result/Qingpu.dat"

    cm, oa, e_acc, miou, fwIoU, kappa = accuracy_analysis(gt_path, res_path)
    print(oa, np.nanmean(e_acc), miou, fwIoU, kappa)

    for i, acc in enumerate(e_acc):
        print(i+1, acc)