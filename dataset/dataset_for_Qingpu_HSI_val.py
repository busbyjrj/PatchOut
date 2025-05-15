import os

import numpy as np
from torch.utils.data import dataset, dataloader


class Qingpu_HSI_Dataset(dataset.Dataset):
    def __init__(self,
                 file_list,
                 training=True,
                 np_seed=2022):
        """
        FCN Full Image Dataset
        """
        self.name = "Qingpu_HSI"
        self.num_classes = 20
        file_list = np.loadtxt(file_list, dtype=str)
        self.name_list = list(file_list[:, 0])
        self.image_list = list(file_list[:, 1])
        self.mask_list = list(file_list[:, 2])

        self.is_training = training

        self._seed = np_seed

    def pre_set(self, image_path, mask_path):
        image = np.load(image_path)  # should be (H, W, C)
        # image = gdal.Open(image_path).ReadAsArray().transpose(1, 2, 0)

        if mask_path is not None:
            mask = np.load(mask_path).squeeze().astype(np.int8)
            # mask = gdal.Open(mask_path).ReadAsArray().squeeze().astype(np.int8)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))

        mask = mask - 1

        im = image.astype(np.float32).transpose(2, 0, 1) # (H, W, C) -> (C, H, W)
        mask = mask

        return im, mask

    def __getitem__(self, idx):
        file_name = self.name_list[idx]
        image_name = self.image_list[idx]
        mask_name = self.mask_list[idx]
        pad_im, pad_mask = self.pre_set(image_name, mask_name)
        return pad_im, pad_mask, file_name

    def __len__(self):
        return len(self.image_list)


def build_Qingpu_HSI_Dataset_for_val(file_list=None, training=False):
    file_list = "./dataset/Qingpu_HSI_val.txt" if file_list is None else file_list
    return Qingpu_HSI_Dataset(file_list, training=training)

