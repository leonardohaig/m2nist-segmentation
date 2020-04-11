#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:Data_provider.py
#       
#Date:20-4-11
#Author:liheng
#Version:V1.0
#============================#

import os
import numpy as np
import tensorflow as tf
import down_data
from multiprocessing import cpu_count

class Data_provider(object):
    def __init__(self,imgs_pth,masks_pth):

        assert os.path.isfile(imgs_pth)
        assert os.path.isfile(masks_pth)

        self.imgs_pth = imgs_pth
        self.masks_pth = masks_pth

        self.input_queue = self._init_dataset()


    def _init_dataset(self):
        imgs = np.load(self.imgs_pth)
        masks = np.load(self.masks_pth)

        self.imgs_num = imgs.shape[0]

        imgs, masks = tf.train.slice_input_producer([imgs,masks],
                                                    capacity=32)

        imgs = tf.cast(imgs,dtype=tf.float32)
        imgs = tf.divide(imgs,255.0) # 归一化[0-1]之间
        masks = tf.cast(masks,dtype=tf.uint8)

        imgs = tf.expand_dims(imgs,axis=-1)

        return [imgs,masks]


    def next_batch(self, batch_size=4):
        #imgs,masks =
        return tf.train.batch(self.input_queue,
                              batch_size=batch_size,
                              num_threads=cpu_count()//2)





if __name__ == '__main__':
    data_rootdir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../','m2nist')
    imgs_pth = os.path.join(data_rootdir, 'train_imgs.npy')
    masks_pth = os.path.join(data_rootdir, 'train_masks.npy')

    dataset = Data_provider(imgs_pth,masks_pth)
    _data = dataset.next_batch(batch_size=2)
    data_batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
        _data,
        num_threads = 2
    )

    with tf.Session() as sess:
        sess.run(tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ))

        cood = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=cood)

        for i in range(5):
        # while True:
            data_queue = sess.run(data_batch_queue.dequeue())
            img = np.reshape(data_queue[0][0],[64,84])
            down_data.show_img_mask(img,data_queue[1][0])
            print(i)


        cood.request_stop()
        cood.join(threads=threads)


