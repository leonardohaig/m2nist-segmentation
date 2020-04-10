#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:down_data.py
#       
#Date:20-4-10
#Author:liheng
#Version:V1.0
#============================#

import numpy as np
import os
import requests
import zipfile
from six.moves import urllib
from tqdm import tqdm
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    file_size = int(urllib.request.urlopen(url).info().get('Content-Length', -1))
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    print(file_size)
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(total=file_size, initial=first_byte, unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with (open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size


def download_m2nist_if_not_exist():
    """

    :return:
    """
    data_rootdir = os.path.join(os.path.split(os.path.realpath(__file__))[0],'m2nist')
    if not os.path.exists(data_rootdir):  # 保存路径不存在，则创建该路径
        os.mkdir(data_rootdir)

    m2nist_zip_path = os.path.join(data_rootdir, 'm2nist.zip')
    if os.path.exists(m2nist_zip_path):
        return
    os.makedirs(data_rootdir, exist_ok=True)
    m2nist_zip_url = 'https://raw.githubusercontent.com/akkaze/datasets/master/m2nist.zip'
    download_from_url(m2nist_zip_url, m2nist_zip_path)
    zipf = zipfile.ZipFile(m2nist_zip_path)
    zipf.extractall(data_rootdir)
    zipf.close()


def show_m2nist(data_rootdir):
    """

    :param data_rootdir:
    :return:
    """
    assert os.path.exists(data_rootdir),data_rootdir + ' path not exist !'

    imgs = np.load(os.path.join(data_rootdir, 'combined.npy')).astype(np.uint8)
    masks = np.load(os.path.join(data_rootdir, 'segmented.npy')).astype(np.uint8)

    for i in range(imgs.shape[0]):
        # 转换为one-hot编码
        mask = to_categorical(masks[i], 11, dtype=np.uint8)

        plt.figure(figsize=(4, 4))

        plt.subplot(4, 4, 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        plt.imshow(imgs[i], cmap=plt.cm.binary)
        plt.xlabel('img' + str(i))

        for idx in range(11):
            plt.subplot(4, 4, idx + 5)
            plt.xticks([])
            plt.yticks([])
            plt.grid('off')
            mask_vis = mask[:, :, idx]
            plt.imshow(mask_vis, cmap=plt.cm.binary)
            plt.xlabel(str(idx))
        plt.show()


if __name__ == '__main__':
    # download_m2nist_if_not_exist()
    data_rootdir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'm2nist')
    show_m2nist(data_rootdir)

    print('Hello world !')
