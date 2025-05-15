# !/usr/bin/env python
# -*-coding:utf-8 -*-

import os
import numpy as np
from osgeo import gdal

def create_image_by_roi(roi_x, roi_y, image_path, save_path, mean=None, std=None):
    src_ds = gdal.Open(image_path)
    if src_ds is None:
        raise ValueError(f"can't open image: {image_path}")

    # create output image
    driver = gdal.GetDriverByName("ENVI")
    dst_ds = driver.Create(
        save_path, 512, 512, src_ds.RasterCount, gdal.GDT_Float32)

    # set simple pixel coordinate system
    dst_geotransform = [roi_x, 1, 0, -roi_y, 0, -1]
    print("GeoTransform of the cropped image:", dst_geotransform)
    dst_ds.SetGeoTransform(dst_geotransform)

    # read and write data
    for i in range(src_ds.RasterCount):
        band = src_ds.GetRasterBand(i + 1)
        data = band.ReadAsArray(roi_x, roi_y, 512, 512)

        # if need to normalize
        if mean is not None and std is not None:
            data = (data - mean[i]) / std[i]

        dst_ds.GetRasterBand(i + 1).WriteArray(data)

    # clean
    dst_ds.FlushCache()
    dst_ds = None
    src_ds = None

    # save as numpy array
    ds = gdal.Open(save_path)
    roi_img = ds.ReadAsArray()
    if roi_img.ndim == 2:
        roi_img = roi_img[np.newaxis, :, :]
    roi_img = roi_img.transpose(1, 2, 0)
    np.save(save_path.replace(".dat", ".npy"), roi_img)

    print("ROI %s is processed." % save_path)


def create_mask_for_testing_by_roi(rois_x, rois_y, gt_path, test_gt_save_path):
    gt_dataset = gdal.Open(gt_path)
    gt_MATRIX = gt_dataset.ReadAsArray()
    print(gt_MATRIX.shape)

    for index in range(len(rois_x)):
        start_col = int(rois_x[index])
        start_row = int(rois_y[index])
        end_row = start_row + 512
        end_col = start_col + 512

        # set gt to 0 in the region
        gt_MATRIX[start_row:end_row, start_col:end_col] = 0

    gt_MATRIX = np.expand_dims(gt_MATRIX, axis=2)
    # save
    driver = gdal.GetDriverByName("ENVI")
    ds = driver.Create(test_gt_save_path, gt_dataset.RasterXSize,
                       gt_dataset.RasterYSize, gt_MATRIX.shape[2], gdal.GDT_Byte)
    for i in range(gt_MATRIX.shape[2]):
        ds.GetRasterBand(i + 1).WriteArray(gt_MATRIX[:, :, i])
    ds.FlushCache()
    ds = None


def sample_labels_all(src_dir, train_dir, test_dir, train_samples):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    src_list = os.listdir(src_dir)

    image_MATRIX_all = []
    fileNames = []
    for file in src_list:
        print(file)
        if not file.endswith(".npy"):
            continue
        src_file = os.path.join(src_dir, file)
        image_MATRIX = np.load(src_file)
        image_MATRIX_all.append(image_MATRIX)
        print(image_MATRIX.shape)
        fileNames.append(file)

    # count the number of pixels of each class in each image
    image_MATRIX_all = np.array(image_MATRIX_all)
    print(image_MATRIX_all.shape)

    # the number of samples to be picked for each class
    train_indices = []
    test_indices = []
    gt_MATRIX = image_MATRIX_all.reshape(-1)
    for i in np.unique(gt_MATRIX):
        indices = np.where(gt_MATRIX == i)[0]
        np.random.shuffle(indices)
        train_indices.extend(indices[:train_samples])
        test_indices.extend(indices[train_samples:])

    # save training samples
    train_data = np.zeros(gt_MATRIX.shape)
    train_data[train_indices] = gt_MATRIX[train_indices]
    train_data = train_data.reshape(
        image_MATRIX_all.shape[0], image_MATRIX_all.shape[1], image_MATRIX_all.shape[2], 1)

    # save validation samples
    test_data = np.zeros(gt_MATRIX.shape)
    test_data[test_indices] = gt_MATRIX[test_indices]
    test_data = test_data.reshape(
        image_MATRIX_all.shape[0], image_MATRIX_all.shape[1], image_MATRIX_all.shape[2], 1)

    print(train_data.shape)
    print(test_data.shape)

    # split training samples for each image
    for index in range(len(fileNames)):
        train_file = os.path.join(
            train_dir, fileNames[index].replace(".npy", ".dat").replace("roi", "train"))
        test_file = os.path.join(
            test_dir, fileNames[index].replace(".npy", ".dat").replace("roi", "val"))

        # save training samples
        MATRIX = train_data[index]
        driver = gdal.GetDriverByName("ENVI")
        ds = driver.Create(
            train_file, MATRIX.shape[1], MATRIX.shape[0], 1, gdal.GDT_Byte)
        for i in range(MATRIX.shape[2]):
            ds.GetRasterBand(i + 1).WriteArray(MATRIX[:, :, i])
        ds = None
        np.save(train_file.replace(".dat", ".npy"), MATRIX)
        print("Train %s is processed." % train_file)

        # save validation samples
        MATRIX = test_data[index]
        driver = gdal.GetDriverByName("ENVI")
        ds = driver.Create(
            test_file, MATRIX.shape[1], MATRIX.shape[0], 1, gdal.GDT_Byte)
        for i in range(MATRIX.shape[2]):
            ds.GetRasterBand(i + 1).WriteArray(MATRIX[:, :, i])
        ds = None
        np.save(test_file.replace(".dat", ".npy"), MATRIX)
        print("Test %s is processed." % test_file)



def create_testing_dataset(src_file, dst_file, mean_std_file):

    mean = np.loadtxt(mean_std_file)[:, 0]
    std = np.loadtxt(mean_std_file)[:, 1]

    src_ds = gdal.Open(src_file)
    x = src_ds.RasterXSize
    y = src_ds.RasterYSize
    band = src_ds.RasterCount

    x_new = x + 256 + 512 - (x + 256) % 512
    y_new = y + 256 + 512 - (y + 256) % 512

    # setting BIP format
    dst_ds = gdal.GetDriverByName('ENVI').Create(dst_file, x_new, y_new, band, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    dst_ds.SetProjection(src_ds.GetProjection())

    # create full data array and write once
    for i in range(band):
        print(i)
        data = src_ds.GetRasterBand(i + 1).ReadAsArray()
        data = (data - mean[i][np.newaxis, np.newaxis]) / std[i][np.newaxis, np.newaxis]
        data = np.pad(data, ((256, 0), (256, 0)), 'symmetric')
        h, w = data.shape
        h_padding = 512 - h % 512
        w_padding = 512 - w % 512
        data = np.pad(data, ((0, h_padding), (0, w_padding)), 'symmetric')
        dst_ds.GetRasterBand(i + 1).WriteArray(data)

    src_ds = None
    dst_ds = None

if __name__ == "__main__":
    src_file = "./Qingpu_HSI_image.dat"
    mean_std_file = "./Qingpu_HSI_mean_std.txt"
    dst_file = "./test/Qingpu_HSI_image_norm_padding.dat"
    create_testing_dataset(src_file, dst_file, mean_std_file)