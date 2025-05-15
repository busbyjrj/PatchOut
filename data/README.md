# Qingpu HSI Dataset Preparation

## Download Instructions
1. Please prepare the Qingpu HSI dataset from [here](https://www.3sobs.cn/qingpu-hsi-dataset)
2. Download the complete dataset archive
3. Extract files to the `data/` directory in this repository
4. Please run `prepare_for_Qingpu_HSI.py`

## File Structure
````
data/
├── Qingpu_HSI_image.dat      # HSI image (ENVI format)
├── Qingpu_HSI_image.hdr      # HSI image (ENVI format)
├── Qingpu_HSI_gt.dat         # HSI label (ENVI format)
├── Qingpu_HSI_gt.hdr         # HSI label (ENVI format)
├── Qingpu_HSI_mean_std.txt
├── Qingpu_HSI_rois.txt
