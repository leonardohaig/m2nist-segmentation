#!/usr/bin/env python3
# coding=utf-8

# ============================#
# Program:utils_torch.py
#       
# Date:20-4-11
# Author:liheng
# Version:V1.0
# ============================#

import yaml
import os
import colorsys
import numpy as np


def get_config(cfg_file: str):
    with open(cfg_file, 'r', encoding='utf-8') as cfg:
        content = cfg.read()
        return yaml.load(content)


# 输入目录路径，输出最新文件完整路径
def find_new_file(dir):
    '''
    查找目录下最新的文件
    '''
    file_lists = os.listdir(dir)
    if not len(file_lists):
        return None

    file_lists.sort(key=lambda fn: os.path.getmtime(os.path.join(dir, fn))
    if not os.path.isdir(os.path.join(dir, fn)) else 0)

    file = os.path.join(dir, file_lists[-1])  # 完整路径

    return file


def tran_masks2images(masks):
    """
    将背景变为黑色，数字0-9用不同颜色显示
    :param masks: [B,H,W],numpy.array 索引0-9 + 背景10
    :return: [B,3,H,W],numpy.array
    """
    masks = masks.astype(np.uint8)

    _masks = np.ones(shape=[masks.shape[0], masks.shape[1], masks.shape[2], 3],
                     dtype=np.uint8) * 10  # [B,H,W,3]

    hsv_tuples = [(1.0 * x / 10, 0.5, 0.8) for x in range(10)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    colors.append((0, 0, 0))  # background

    for i in range(11):
        _masks[np.where(masks == i)] = colors[i]

    return _masks.transpose(0, 3, 1, 2)  # [B,3,H,W]
