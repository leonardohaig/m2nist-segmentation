#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:predict.py
#       
#Date:20-4-12
#Author:liheng
#Version:V1.0
#============================#

import model
import tensorflow as tf
import argparse
import os
import numpy as np
import time
import cv2

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_pth',type=str,help='The images path',
                        default='../m2nist/combined.npy')
    parser.add_argument('--weights_pth',type=str,help='The model weights path',
                        default='./checkpoint/m2nist_model_4-epoch.ckpt-2500')

    return parser.parse_args()


def test_model(imgs_pth,weights_pth):
    """

    :param imgs_pth:
    :param weights_pth:
    :return:
    """

    input_data = tf.placeholder(dtype=tf.float32,shape=[1,64,84,1],name='input_data')
    net = model.Model(input_data,tf.constant(False,dtype=tf.bool))
    decode_ret = tf.argmax(net.get_inference(),axis=-1)

    imgs = np.load(imgs_pth)

    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))
        tf.train.Saver(tf.global_variables()).restore(sess,weights_pth)

        nWaitTime = 0
        img_idx = -1

        for img in imgs:
            prev_time = time.time()
            img_idx += 1


            img = np.expand_dims(img,axis=-1).astype(np.float32)/255.0
            seg_img = sess.run(decode_ret,feed_dict={input_data:img[np.newaxis,...]})
            seg_img = seg_img[0][...,np.newaxis]
            seg_img[seg_img<10] = 255
            seg_img[seg_img==11] = 0
            seg_img = seg_img.astype(np.float32)/255

            exec_time = time.time()-prev_time # s

            seg_img = np.concatenate([img,seg_img],axis=1)
            # cv2.imshow('ori_img',img)
            cv2.imshow('res_img',seg_img)
            key = cv2.waitKey(nWaitTime)
            if 27 == key:  # ESC
                break
            elif 32 == key:  # space
                nWaitTime = not nWaitTime


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    # init args
    args = init_args()

    imgs_pth = args.imgs_pth
    weights_pth = args.weights_pth

    assert os.path.isfile(imgs_pth), 'there is no images file !'
    assert os.path.isfile(weights_pth+'.meta'), 'there is no weights file !'


    test_model(imgs_pth,weights_pth)