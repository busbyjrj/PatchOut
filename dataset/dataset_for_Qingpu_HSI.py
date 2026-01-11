import os
import time
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import dataset, dataloader
from torchvision import transforms
from osgeo import gdal


class Qingpu_HSI_Dataset_subminibatch(dataset.Dataset):
    """
    FCN Full Image Dataset
    Some of the codes refer to FreeNet https://github.com/Z-Zheng/FreeNet
    """

    def __init__(self,
                 file_list,
                 training=True,
                 sub_minibatch=0.5,
                 train_percent=1.0,
                 multiplier=6,
                 np_seed=2022):
        self.name = "Qingpu_HSI"
        self.num_classes = 20
        file_list = np.loadtxt(file_list, dtype=str)
        np.random.shuffle(file_list)
        self.name_list = list(file_list[:, 0])
        self.image_list = list(file_list[:, 1])
        self.mask_list = list(file_list[:, 2])
        self.len = len(self.image_list)

        self.is_training = training
        if self.is_training:
            self.len = self.len * multiplier

        self.sub_minibatch = sub_minibatch
        self.train_percent = train_percent

        # set list lenght = 9999 to make sure seeds enough
        # 2 << 31 -1  ==  2**31
        self._seed = np_seed
        self._rs = np.random.RandomState(self._seed)
        self.seeds_for_minibatchsample = [
            e for e in self._rs.randint(low=2 << 31 - 1, size=9999)]

    def __getitem__(self, idx):
        if self.is_training:
            idx = idx % len(self.image_list)
            file_name = self.name_list[idx]
            image_name = self.image_list[idx]
            mask_name = self.mask_list[idx]
            pad_im, pad_mask, train_indicator, test_indicator = self.pre_set(
                image_name, mask_name)
            train_inds_list = self.minibatch_sample(pad_mask, train_indicator,
                                                    self.sub_minibatch,
                                                    self.seeds_for_minibatchsample.pop())
            return pad_im, pad_mask, train_inds_list, file_name
        else:
            file_name = self.name_list[idx]
            image_name = self.image_list[idx]
            mask_name = self.mask_list[idx]
            pad_im, pad_mask, train_indicator, test_indicator = self.pre_set(
                image_name, mask_name)
            return pad_im, pad_mask, test_indicator, file_name

    def __len__(self):
        return self.len

    def pad_image_mirror(self, image, mask):
        pad_size = 128
        pad_im = np.pad(
            image, ((pad_size, 0), (pad_size, 0), (0, 0)), 'symmetric')
        pad_mask = np.pad(mask, ((pad_size, 0), (pad_size, 0)), 'symmetric')
        pad_im = np.pad(
            pad_im, ((0, pad_size), (0, pad_size), (0, 0)), 'symmetric')
        pad_mask = np.pad(
            pad_mask, ((0, pad_size), (0, pad_size)), 'symmetric')
        return pad_im, pad_mask

    def pre_set(self, image_path, mask_path):
        # .transpose(1, 2, 0)  # should be (H, W, C)
        image = np.load(image_path)
        # image = gdal.Open(image_path).ReadAsArray().transpose(1, 2, 0)

        if mask_path is not None:
            mask = np.load(mask_path).squeeze().astype(np.int8)
            # mask = gdal.Open(mask_path).ReadAsArray().squeeze().astype(np.int8)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))

        mask = mask - 1

        image, mask = self.pad_image_mirror(image, mask)
        image, mask = self.sub_patch(image, mask, int(time.time()))
        image, mask = self.random_rotation(image, mask)

        train_indicator, test_indicator = self.fixed_num_sample(mask,
                                                                self.num_classes,
                                                                self.train_percent,
                                                                self._seed)

        im = image.transpose(2, 0, 1).astype(np.float32)  # (H, W, C) -> (C, H, W)
        mask = mask.astype(np.float32)
        train_indicator = train_indicator.astype(np.float32)
        test_indicator = test_indicator.astype(np.float32)

        return im, mask, train_indicator, test_indicator

    def random_rotation(self, im, mask):
        # augmentation dataset for HSI
        angle = np.random.randint(0, 4)
        im = np.rot90(im, angle, (0, 1))
        mask = np.rot90(mask, angle, (0, 1))

        if np.random.random() > 0.5:
            im = np.flip(im, axis=0)
            mask = np.flip(mask, axis=0)

        im = np.ascontiguousarray(im)
        mask = np.ascontiguousarray(mask)
        return im, mask

    def fixed_num_sample(self, gt_mask: np.ndarray, num_classes, train_percent, seed=2023):
        """
        Args:
            gt_mask: 2-D array of shape [height, width]
            num_train_samples: int
            num_classes: scalar
            seed: int

        Returns:
            train_indicator, test_indicator
        """
        rs = np.random.RandomState(seed)
        gt_mask_flatten = gt_mask.ravel()
        train_indicator = np.zeros_like(gt_mask_flatten)
        test_indicator = np.zeros_like(gt_mask_flatten)

        for cls in range(0, num_classes):
            inds = np.where(gt_mask_flatten == cls)[0]
            rs.shuffle(inds)
            num_train_samples = int(len(inds) * train_percent)
            train_inds = inds[:num_train_samples]
            test_inds = inds[num_train_samples:]
            train_indicator[train_inds] = 1
            test_indicator[test_inds] = 1

        train_indicator = train_indicator.reshape(gt_mask.shape)
        test_indicator = test_indicator.reshape(gt_mask.shape)

        return train_indicator, test_indicator

    # 有了训练点位置，进行minibatch
    def minibatch_sample(self, gt_mask: np.ndarray, train_indicator: np.ndarray, minibatch_size, seed):
        """
        Args:
            gt_mask: 2-D array of shape [height, width]
            train_indicator: 2-D array of shape [height, width]
            minibatch_size:
        Returns:
        """
        rs = np.random.RandomState(seed)
        # split into N classes
        cls_list = np.unique(gt_mask)
        inds_dict_per_class = dict()
        for cls in cls_list:
            train_inds_per_class = np.where(
                gt_mask == cls, train_indicator, np.zeros_like(train_indicator))
            inds = np.where(train_inds_per_class.ravel() == 1)[0]
            rs.shuffle(inds)
            inds_dict_per_class[cls] = inds

        train_inds_list = []
        cnt = 0
        while True:
            train_inds = np.zeros_like(train_indicator).ravel()
            for cls, inds in inds_dict_per_class.items():
                minibatch_size_per_class = int(
                    np.ceil(minibatch_size * len(inds)))
                left = cnt * minibatch_size_per_class
                if left >= len(inds):
                    continue
                # remain last batch though the real size is smaller than minibatch_size
                right = min((cnt + 1) * minibatch_size_per_class, len(inds))
                fetch_inds = inds[left:right]
                train_inds[fetch_inds] = 1
            cnt += 1
            if train_inds.sum() == 0: 
                return train_inds_list
            train_inds_list.append(train_inds.reshape(train_indicator.shape))

    def sub_patch(self, image, mask, seed):
        height, width, _ = image.shape
        patch_size = 512
        np.random.seed(seed)
        x = np.random.randint(0, height - patch_size)
        y = np.random.randint(0, width - patch_size)
        sub_image = image[x:x + patch_size, y:y + patch_size, :]
        sub_mask = mask[x:x + patch_size, y:y + patch_size]
        return sub_image, sub_mask


def build_Qingpu_HSI_dataset(file_list=None, training=True):
    if training:
        file_list = "./dataset/Qingpu_HSI_train.txt" if file_list is None else file_list
        dataset = Qingpu_HSI_Dataset_subminibatch(file_list, training=True, sub_minibatch=0.5,
                                                  train_percent=1.0, np_seed=2022)
    return dataset

