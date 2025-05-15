# !/usr/bin/env python
# -*-coding:utf-8 -*-

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from osgeo import gdal
from torch import optim
from tqdm import tqdm
from dataset.sliding_window_dataset import SlidingWindowDataset
from utils.accuracy_analysis import accuracy_analysis
import random
from datetime import datetime
import argparse
import logging
import sys
import time
import numpy as np
from timm.data.loader import MultiEpochsDataLoader

import torch
import torch.backends.cudnn as cudnn
from thop import profile
from dataset.dataset_for_Qingpu_HSI import build_Qingpu_HSI_dataset
from dataset.dataset_for_Qingpu_HSI_val import build_Qingpu_HSI_Dataset_for_val
from model.PatchOut import PatchOut
from ops_FCN_Sample import FCN_train_one_epoch, FCN_test_one_epoch
from utils.utils import save_checkpoint, count_parameters_in_MB, metrics, load_pretrained_model


parser = argparse.ArgumentParser(description='train base net')
parser.add_argument('--net_name', default='PatchOut', help='network name')
parser.add_argument('--save_path', type=str, default='./checkpoints', help='models and logs are saved here')
parser.add_argument('--data_path', type=str, default='./datasets', help='path name of image dataset')
parser.add_argument('--dataset_name', type=str, default='Qingpu_HSI', help='dataset name')
parser.add_argument('--in_channels', type=int, default=251, help='input channels')
parser.add_argument('--image_size', type=int, default=512, help='image size')
parser.add_argument('--num_classes', type=int, default=20, help='number of classes')
parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=4, help='The size of batch')
parser.add_argument('--use_cuda', type=int, default=True, help='use cuda or not')
parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument('--print_freq', type=int, default=4, help='frequency of showing training results on console')
parser.add_argument('--checkpoint_path', type=int, default=None, help='is pretrained or not')
parser.add_argument('--is_draw', type=bool, default=True, help='draw gt and prediction or not')

# setting
args, unparsed = parser.parse_known_args()

# cuda
if torch.cuda.is_available() and args.use_cuda:
    use_cuda = True
else:
    use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
print("using GPU:", use_cuda)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.enabled = True
cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# time and save path
start_time = time.time()
print("start time:", datetime.utcfromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"))
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# logging file
log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("args = %s", args)
logging.info("unparsed_args = %s", unparsed)

# platform
import platform

if platform.system().lower() == 'windows':
    logging.info("windows")
    num_workers = 0
elif platform.system().lower() == 'linux':
    logging.info("linux")
    num_workers = min(args.batch_size * 4, os.cpu_count())
else:
    num_workers = 0

# Network
logging.info('----------- Network Initialization --------------')

net = PatchOut(in_channels=args.in_channels, num_classes=args.num_classes,
                    image_size=512, reduce_ratio=4)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
scheduler = None
net.to(device)
logging.info('%s' % net)
img = torch.randn(1, args.in_channels, args.image_size, args.image_size).to(device)
flops, params = profile(net, inputs=(img,))
logging.info("param size = %f MB" % (params / 1000 / 1000.0))
logging.info("flops size = %f MB" % (flops / 1000 / 1000.0))
logging.info('-----------------------------------------------')
# net = torch.compile(net) # after torch 2.0

if args.checkpoint_path is not None:
    print("Loading the pretrained model......")
    print(args.checkpoint_path)
    checkpoint = torch.load(args.checkpoint_path)
    net = load_pretrained_model(net, checkpoint['net'])

# define loss function
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
criterion = criterion.to(device)

# create HSI dataset
train_dataset = build_Qingpu_HSI_dataset(training=True)
val_dataset = build_Qingpu_HSI_Dataset_for_val(training=False)

args.num_classes = train_dataset.num_classes

# # load HSI dataset
# train_loader = data.DataLoader(dataset=train_dataset,
#                                batch_size=args.batch_size,
#                                shuffle=True,
#                                num_workers=num_workers,
#                                persistent_workers=True,
#                                pin_memory=True)
# val_loader = data.DataLoader(dataset=val_dataset,
#                              batch_size=args.batch_size,
#                              shuffle=False,
#                              num_workers=num_workers,
#                              persistent_workers=True,
#                              pin_memory=True)

train_loader = MultiEpochsDataLoader(dataset=train_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     pin_memory=True)
val_loader = MultiEpochsDataLoader(dataset=val_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   pin_memory=True)

best_top1 = 1e9
for epoch in (range(1, args.epochs + 1)):
    epoch_start_time = time.time()
    logging.info('Epoch: %d/%d', epoch, args.epochs)

    # train one epoch
    epoch_time, losses = FCN_train_one_epoch(train_loader, net, criterion, optimizer, epoch, device, logging)
    logging.info("train loss: " + str(losses))

    if epoch % 5 == 0:
        # evaluate on test set
        logging.info("val the models...")
        losses = FCN_test_one_epoch(val_loader, net, criterion, device, logging)
        logging.info("val loss: " + str(losses))

        # save the best model
        is_best = False
        if losses < best_top1:
            best_top1 = losses
            is_best = True
        logging.info('Saving models......')
        state = {'epoch': epoch,
                 'net': net.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'prec1': losses,
                 'seed': args.seed}
        save_checkpoint(state, is_best, args.save_path)

end_time = time.time()
logging.info("end time:" + datetime.utcfromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"))
logging.info("total time:" + str(end_time - start_time))

if args.is_draw:
    # load pretrained model
    logging.info('Loading the best model......')
    checkpoint = torch.load(os.path.join(args.save_path, "model_best.pth.tar"))
    net = load_pretrained_model(net, checkpoint['net'])
    net.eval()
  
    gt_path = r"./data/test/Qingpu_HSI_test_gt.dat"
    image_path = r"./data/test/Qingpu_HSI_image_norm_padding.dat"
    grid = (512, 512)
    stride = (256, 256)
    Dataset = SlidingWindowDataset(image_path, grid, stride, preload=True)

    # test
    logging.info('----------- Test --------------')
    predictions = np.zeros((args.num_classes, Dataset.H, Dataset.W))
    for i in tqdm(range(len(Dataset))):
        image, i, j = Dataset[i]
        image = torch.from_numpy(image).to(device)
        with torch.no_grad():
            output = net(image.unsqueeze(0))
            output = output.cpu().numpy()
            output = output[0]
            predictions[:, i:i + grid[0], j:j + grid[1]] += output

    src_ds = gdal.Open(gt_path)
    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize
    prediction = np.argmax(predictions, axis=0) + 1
    start_x, start_y = 256, 256
    end_x, end_y = cols + 256, rows + 256
    class_matrix = prediction[start_y:end_y, start_x:end_x]

    # save
    save_path = r"./result/" + args.dataset_name + ".dat"
    dst_ds = gdal.GetDriverByName('ENVI').Create(save_path, cols, rows, 1, gdal.GDT_Int16)
    dst_ds.GetRasterBand(1).WriteArray(class_matrix)
    dst_ds.FlushCache()
    dst_ds = None
    src_ds = None

    logging.info("test time:" + str(end_time - start_time))

    # accuracy analysis
    cm, oa, e_acc, miou, fwIoU, kappa = accuracy_analysis(gt_path, save_path)
    m_aa = np.mean(e_acc)

    logging.info("OA: " + str(oa))
    logging.info("AA: " + str(e_acc))
    logging.info("mAA: " + str(m_aa))
    logging.info("MIoU: " + str(miou))
    logging.info("FWIoU: " + str(fwIoU))
    logging.info("Kappa: " + str(kappa))