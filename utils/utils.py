# !/usr/bin/env python
# -*-coding:utf-8 -*-


import os
import shutil

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_test(output, target, topk=(1,), ignored_labels=None):
    """Computes the precision@k for the specified values of k"""
    if ignored_labels is None:
        ignored_labels = []
    # ignored_mask = np.zeros(target.shape[:2], dtype=bool)
    ignored_mask = torch.zeros(target.shape[:2], dtype=torch.bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask

    target = target[ignored_mask]
    output = output[ignored_mask]

    maxk = max(topk)
    if len(target.shape) == 1:
        batch_size = 1
    else:
        batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def metrics(prediction, target, ignored_labels=None, n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """

    if ignored_labels is None:
        ignored_labels = []
    ignored_mask = np.zeros(target.shape[:2], dtype=bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask

    if ignored_mask.shape[0] != prediction.shape[0]:
        cha = ignored_mask.shape[0] - prediction.shape[0]
        ignored_mask = ignored_mask[int(cha / 2) - 1:-(int(cha / 2) + 1), int(cha / 2) - 1:-(int(cha / 2) + 1)]
        target = target[int(cha / 2) - 1:-(int(cha / 2) + 1), int(cha / 2) - 1:-(int(cha / 2) + 1)]

    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target - 1,
        prediction - 1,
        labels=range(n_classes))

    results["Confusion matrix"] = cm
    results["E ACC"] = [cm[x][x] / np.sum(cm[x, :]) for x in range(len(cm))]

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["OA"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1 scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    # Compute the average precision and recall score by renjie
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    for i in range(n_classes):
        precision[i] = cm[i, i] / np.sum(cm[i, :])
        recall[i] = cm[i, i] / np.sum(cm[:, i])
    average_precision = np.mean(precision)
    average_recall = np.mean(recall)
    results["Average precision"] = average_precision
    results["Average recall"] = average_recall

    return results


def save_checkpoint(state, is_best, save_path):
    epoch = state['epoch']
    file_path = os.path.join(save_path, "epoch_{}.pth".format(epoch))
    torch.save(state, file_path)
    if is_best:
        best_save_path = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(file_path, best_save_path)


def count_parameters_in_MB(model):
    return sum(p.numel() for p in model.parameters()) / 1000000.0


def load_pretrained_model(model, pretrained_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k.replace("_orig_mod.", ""): v for k, v in pretrained_dict.items()}

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    return model


def remove_compiled_prefix(checkpoint):
    keys_list = list(checkpoint['net'].keys())
    for key in keys_list:
        if 'orig_mod.' in key:
            deal_key = key.replace('_orig_mod.', '')
            checkpoint['net'][deal_key] = checkpoint['net'][key]
            del checkpoint['net'][key]
    return checkpoint
