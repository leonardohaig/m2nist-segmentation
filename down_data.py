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
import matplotlib.pyplot as plt


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.

    # Example

    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


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

def show_img_mask(img,mask):
    mask = to_categorical(mask,11,dtype=np.uint8)

    plt.figure(figsize=(4, 4))

    plt.subplot(4, 4, 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(img, cmap='Greys_r')
    plt.xlabel('img')

    for idx in range(11):
        plt.subplot(4, 4, idx + 5)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        mask_vis = mask[:, :, idx]
        plt.imshow(mask_vis, cmap='Greys_r')
        plt.xlabel(str(idx))
    # plt.get_current_fig_manager().full_screen_toggle()
    plt.show()


def show_m2nist(imgs_pth,masks_pth):
    """

    :param data_rootdir:
    :return:
    """
    assert os.path.isfile(imgs_pth),imgs_pth + ' path not exist !'
    assert os.path.isfile(masks_pth), masks_pth + ' path not exist !'

    imgs = np.load(imgs_pth).astype(np.uint8)
    masks = np.load(masks_pth).astype(np.uint8)

    for i in range(imgs.shape[0]):
        # 转换为one-hot编码
        mask = to_categorical(masks[i], 11, dtype=np.uint8)

        plt.figure(figsize=(4, 4))

        plt.subplot(4, 4, 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        plt.imshow(imgs[i], cmap='Greys_r')
        plt.xlabel('img' + str(i))

        for idx in range(11):
            plt.subplot(4, 4, idx + 5)
            plt.xticks([])
            plt.yticks([])
            plt.grid('off')
            mask_vis = mask[:, :, idx]
            plt.imshow(mask_vis, cmap='Greys_r')
            plt.xlabel(str(idx))
        # plt.get_current_fig_manager().full_screen_toggle()
        plt.show()


def split_m2nist(data_rootdir):
    """

    :param data_rootdir:
    :return:
    """

    assert os.path.exists(data_rootdir), data_rootdir + ' path not exist !'

    imgs = np.load(os.path.join(data_rootdir, 'combined.npy'))
    masks = np.load(os.path.join(data_rootdir, 'segmented.npy'))

    val_ratio = 0.2
    num_data = imgs.shape[0]
    num_train = int(num_data * (1 - val_ratio))

    train_imgs_pth = os.path.join(data_rootdir, 'train_imgs.npy')
    train_masks_pth = os.path.join(data_rootdir,'train_masks.npy')
    val_imgs_pth = os.path.join(data_rootdir,'val_imgs.npy')
    val_masks_pth = os.path.join(data_rootdir, 'val_masks.npy')



    np.save(train_imgs_pth,imgs[:num_train,...])
    np.save(train_masks_pth,masks[:num_train,...])

    np.save(val_imgs_pth,imgs[num_train:,...])
    np.save(val_masks_pth,masks[num_train:,...])




if __name__ == '__main__':
    data_rootdir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'm2nist')
    # download_m2nist_if_not_exist()
    # split_m2nist(data_rootdir)

    imgs_pth = os.path.join(data_rootdir,'train_imgs.npy')
    masks_pth = os.path.join(data_rootdir,'train_masks.npy')
    show_m2nist(imgs_pth,masks_pth)

    print('Hello world !')
