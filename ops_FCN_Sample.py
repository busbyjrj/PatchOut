# !/usr/bin/env python
# -*-coding:utf-8 -*-

import sys
import time
import torch
from tqdm import tqdm
from utils.utils import AverageMeter

print_freq = 10


def FCN_train_one_epoch(train_loader, net, criterion, optimizer, epoch, device, logging):
    losses = AverageMeter()
    net.train()
    start_time = time.time()

    # training
    for index, batch in enumerate(train_loader, start=1):
        image, label, weight_list, filename = batch
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        for i in range(len(weight_list)):
            optimizer.zero_grad()
            weight = weight_list[i].to(device, non_blocking=True)

            res = net(image)
            label_iter = torch.where(weight == 1, label, torch.zeros_like(label) - 1).type(torch.cuda.LongTensor)
            loss = criterion(res, label_iter)

            loss.backward()
            optimizer.step()

            losses.update(loss.item(), image.size(0))

        log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
                   'loss:{losses.val:.4f}({losses.avg:.4f}) ').format(
            epoch, index, len(train_loader), losses=losses)
        logging.info(log_str)

    end_time = time.time()
    epoch_time = end_time - start_time
    log_str = ('Epoch[{0}] '
               'Epoch time:{time:.2f} '
               'loss:{losses.avg:.4f} ').format(
        epoch, time=epoch_time, losses=losses)
    logging.info(log_str)

    return epoch_time, losses.avg


@torch.no_grad()
def FCN_test_one_epoch(test_loader, net, criterion, device, logging):
    losses = AverageMeter()
    net.eval()

    test_loader = tqdm(test_loader, file=sys.stdout)

    for index, batch in enumerate(test_loader, start=1):
        image, label, filename = batch
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True).type(torch.cuda.LongTensor)

        out = net(image)
        loss = criterion(out, label)

        losses.update(loss.item(), image.size(0))

    logging.info('Loss: {:.4f}'.format(losses.avg))

    return losses.avg
