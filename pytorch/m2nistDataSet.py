#!/usr/bin/env python3
# coding=utf-8

# ============================#
# Program:m2nistDataSet.py
#       数据加载模块
# Date:20-4-16
# Author:liheng
# Version:V1.0
# ============================#

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import utils_torch
from multiprocessing import cpu_count

__all__ = ['m2nistDataLoader']


class m2nistDatase(Dataset):
    """

    """

    def __init__(self, imgs_pth, masks_pth):
        assert os.path.isfile(imgs_pth)
        assert os.path.isfile(masks_pth)

        # load
        imgs = np.load(imgs_pth)
        masks = np.load(masks_pth)

        # padding
        # 从[64,84]填充大小到[64,96],
        # 对于图像采用0填充；对于label采用10常值填充，因为10代表背景
        imgs = np.pad(imgs, ((0, 0), (0, 0), (6, 6)), 'constant', constant_values=0)
        masks = np.pad(masks, ((0, 0), (0, 0), (6, 6)), 'constant', constant_values=10)

        self.imgs = np.expand_dims(imgs.astype(np.float32) / 255, axis=1)  # [B,C,H,W]
        self.masks = masks.astype(np.uint8)

    def __getitem__(self, index):
        img = torch.tensor(self.imgs[index])
        mask = torch.tensor(self.masks[index])

        return img, mask

    def __len__(self):
        return self.imgs.shape[0]


def m2nistDataLoader(cfg_pth, dataset_type='train'):
    """

    :param cfg_pth:
    :param dataset_type: train  or val (验证集validation)
    :return:
    """

    assert os.path.isfile(cfg_pth), 'config file does not exist !'
    config = utils_torch.get_config(cfg_pth)

    if dataset_type == 'train':
        imgs_pth = config['Train.images_pth']
        masks_pth = config['Train.masks_pth']
        batch_size = config['Train.batch_size']
    else:
        imgs_pth = config['Val.images_pth']
        masks_pth = config['Val.masks_pth']
        batch_size = config['Val.batch_size']

    dataset = m2nistDatase(imgs_pth, masks_pth)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=cpu_count() // 2)

    return dataloader


if __name__ == '__main__':
    os.chdir(os.path.split(os.path.abspath(__file__))[0])

    sys.path.append('..')
    import down_data

    if 0:
        data_rootdir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../', 'm2nist')
        imgs_pth = os.path.join(data_rootdir, 'train_imgs.npy')
        masks_pth = os.path.join(data_rootdir, 'train_masks.npy')

        dataset = m2nistDatase(imgs_pth, masks_pth)
        dataloader = DataLoader(dataset=dataset, batch_size=6,
                                shuffle=True, num_workers=cpu_count() // 2)
    else:
        dataloader = m2nistDataLoader('./config.yaml')

    for i, img_mask in enumerate(dataloader):
        img = np.squeeze(img_mask[0][0].numpy())
        down_data.show_img_mask(img, img_mask[1][0].numpy())
