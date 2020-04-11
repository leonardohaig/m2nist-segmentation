#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:common.py
#       定义yolo v3中的卷积层、残差层以及张量拼接和上采样函数
#Date:2019.05.20
#Author:liheng
#Version:V1.0
#Reference:https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/common.py
#============================#

__author__ = 'liheng'

import tensorflow as tf
import math


def batch_normalization(input_data, training, decay=0.9):
    """
    :param input_data: format is 'NHWC'
    :param training: 是否在训练，即bn会根据该参数选择mean and variance
    :param decay: 均值方差滑动参数
    :return: BN后的数据
    """
    with tf.variable_scope('BatchNorm'):
        input_c = input_data.get_shape().as_list()[-1] # channel of input_data
        gamma = tf.get_variable(name='gamma', shape=input_c, dtype=tf.float32,
                                initializer=tf.ones_initializer, trainable=True)
        beta = tf.get_variable(name='beta', shape=input_c, dtype=tf.float32,
                               initializer=tf.zeros_initializer, trainable=True)
        moving_mean = tf.get_variable(name='moving_mean', shape=input_c, dtype=tf.float32,
                                      initializer=tf.zeros_initializer, trainable=False)
        moving_variance = tf.get_variable(name='moving_variance', shape=input_c, dtype=tf.float32,
                                          initializer=tf.ones_initializer, trainable=False)

        def mean_and_var_update():
            axes = (0, 1, 2)
            batch_mean = tf.reduce_mean(input_data, axis=axes)
            batch_var = tf.reduce_mean(tf.pow(input_data - batch_mean, 2), axis=axes)
            with tf.control_dependencies([tf.assign(moving_mean, moving_mean * decay + batch_mean * (1 - decay)),
                                          tf.assign(moving_variance, moving_variance * decay + batch_var * (1 - decay))]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, variance = tf.cond(training, mean_and_var_update, lambda: (moving_mean, moving_variance))
        return tf.nn.batch_normalization(input_data, mean, variance, beta, gamma, 1e-05)

def group_normalization(input_data, input_c, num_group=32, eps=1e-5):
    """
    :param input_data: format is 'NHWC'，C必须是num_group的整数倍
    :param input_c: channel of input_data
    :return: GN后的数据
    """
    with tf.variable_scope('GroupNorm'):
        input_shape = tf.shape(input_data)
        N = input_shape[0]
        H = input_shape[1]
        W = input_shape[2]
        C = input_c
        assert (C % num_group) == 0
        input_data = tf.reshape(input_data, (N, H, W, num_group, C // num_group))
        axes = (1, 2, 4)
        mean = tf.reduce_mean(input_data, axis=axes, keep_dims=True)
        std = tf.sqrt(tf.reduce_mean(tf.pow(input_data - mean, 2), axis=axes, keep_dims=True) + eps)
        input_data = 1.0 * (input_data - mean) / std
        input_data = tf.reshape(input_data, (N, H, W, C))
        gamma = tf.get_variable(name='gamma', shape=C, dtype=tf.float32,
                                initializer=tf.ones_initializer, trainable=True)
        beta = tf.get_variable(name='beta', shape=C, dtype=tf.float32,
                               initializer=tf.zeros_initializer, trainable=True)
    return gamma * input_data + beta

def convolutional(input_data, filters_shape,
                  trainable,name,
                  downsample=False,activate=True,bn=True):
    '''
    yolo v3 中的 卷积层=卷积+BN+LeakyReLU
    :param input_data:输入的批数据 [batch, in_height, in_width, in_channels]
    :param filters_shape:卷积核 [filter_height, filter_width, in_channels, out_channels]
    :param trainable:训练or验证。作用于批规范化选项，训练时，为True，验证时为False
    :param name:卷积操作名称（限定作用域，主要是用来限定weight和bias的作用域）
    :param downsample:下采样方式
    :param activate:对输出是否采用激活函数进行激活
    :param bn:是否批规范化
    :return: feature map [batch, height, width, channels]
    '''
    with tf.variable_scope(name):
        if downsample:
            pad_h,pad_w = (filters_shape[0]-2)//2+1,(filters_shape[1]-2)//2+1#填充高，宽
            paddings = tf.constant([[0,0],[pad_h,pad_h],[pad_w,pad_w],[0,0]])
            input_data = tf.pad(input_data,paddings,'CONSTANT')
            strides = (1,2,2,1)
            padding = 'VALID'
        else:
            strides = (1,1,1,1)
            padding = 'SAME'

        # 获取变量值(如果该变量不存在则创建该变量，下次直接获取变量值，类似于c++中的static变量，后面的bias同理)
        weight = tf.get_variable(name='weight',dtype=tf.float32,trainable=True,
                                 shape=filters_shape,initializer=tf.random_normal_initializer(stddev=0.01))

        #conv2d参数说明：
        #第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，
        #   具有[batch, in_height, in_width, in_channels]这样的shape，
        #   具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数(or 厚度)]，注意这是一个4维的Tensor，
        #   要求类型为float32和float64其中之一
        #第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，
        #   具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
        #   具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，
        #   有一个地方需要注意，第三维in_channels，就是参数input的第四维
        #第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4。
        #   对于图片，因为只有两维，通常strides取[1，stride，stride，1]
        #第四个参数padding：string类型的量，只能是"SAME", "VALID"其中之一，这个值决定了不同的卷积方式
        #第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
        #结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。
        conv = tf.nn.conv2d(input=input_data,filter=weight,strides=strides,padding=padding)

        if bn:
            #批规范化
            #原理公式：y=γ(x-μ)/σ+β.其中x是输入，y是输出，μ是均值，σ是方差，γ和β是缩放（scale）、偏移（offset）系数
            #       μ和σ，在训练的时候，使用的是batch内的统计值，测试/预测的时候，采用的是训练时计算出的滑动平均值
            #总的来说就是对于同一batch的input，假设输入大小为[batch_num, height, width, channel]，
            #   逐channel地计算同一batch中所有数据的mean和variance，
            #   再对input使用mean和variance进行归一化，最后的输出再进行线性平移，得到batch_norm的最终结果。
            #伪代码如下(https://blog.csdn.net/huitailangyz/article/details/85015611)：
            '''
            for i in range(channel):
                x = input[:, :, :, i]
                mean = mean(x)
                variance = variance(x)
                x = (x - mean) / sqrt(variance)
                x = scale * x + offset
                input[:, :, :, i] = x

            '''
            # conv = tf.layers.batch_normalization(conv,beta_initializer=tf.zeros_initializer(),
            #                                      gamma_initializer=tf.ones_initializer(),
            #                                      moving_mean_initializer=tf.zeros_initializer(),
            #                                      moving_variance_initializer=tf.ones_initializer(),
            #                                      training=trainable)#BN操作，训练时training=True，测试时training=False
            conv = batch_normalization(input_data=conv, training=trainable)#BN操作，训练时training=True，测试时training=False
            # conv = group_normalization(input_data=conv,input_c=conv.get_shape().as_list()[-1])
        else:
            bias = tf.get_variable(name='bias',shape=filters_shape[-1],trainable=True,
                                   dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv,bias)

        if activate:
            conv = tf.nn.relu6(conv)#激活函数

    return conv

def residual_block(input_data,input_channel,filter_num1,filter_num2,trainable,name):
    '''
    残差模块，残差输出 = 输入 + (1X1convolutional输入 + 3X3convolutional的1X1输出)
    :param input_data:
    :param input_channel:
    :param filter_num1:
    :param filter_num2:输出厚度
    :param trainable:
    :param name:
    :return:
    '''
    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data,filters_shape=(1,1,input_channel,filter_num1),
                                   trainable=trainable,name='conv1')# 1X1卷积操作

        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1, filter_num2),
                                   trainable=trainable, name='conv2')# 3X3卷积操作

        residual_output = short_cut + input_data

    return residual_output

def sep_residual_block(input_data,input_channel,filter_num1,filter_num2,trainable,name):
    '''
    深度可分离卷积实现残差模块，残差输出 = 输入 + (1X1convolutional输入 + 3X3convolutional的1X1输出)
    :param input_data:
    :param input_channel:
    :param filter_num1:
    :param filter_num2:输出厚度
    :param trainable:
    :param name:
    :return:
    '''
    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data,filters_shape=(1,1,input_channel,filter_num1),
                                   trainable=trainable,name='conv1')# 1X1卷积操作
        input_data = separable_conv(input_data,filter_num2,
                                    training=trainable,
                                    name='conv2',downsample=False)# 3X3深度可分离卷积操作

        residual_output = short_cut + input_data

    return residual_output


def route(name,previous_output,current_output):
    '''
    张量拼接操作
    :param name:
    :param previous_output:
    :param current_output:
    :return:
    '''

    with tf.variable_scope(name):
        output = tf.concat([previous_output,current_output],axis=-1)#对-1的理解：axis=-1表示对张量的最后一维进行拼接
    return output

def upsample(input_data,name,method='deconv'):
    '''
    上采样操作
    :param input_data:
    :param name:
    :param method:
    :return:
    '''
    assert method in ['resize','deconv']

    if method == 'resize':
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(images=input_data,size=(input_shape[1]*2,input_shape[2]*2))
            #images: A Tensor.4-D with shape [batch, height, width, channels]
            #size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new size for the images.
            #return: A Tensor. Has the same type as images.

    if method == 'deconv':
        num_filter = input_data.get_shape().as_list()[-1]#TODO:该行代码有疑问
        #反卷积操作or转置卷积
        output = tf.layers.conv2d_transpose(input_data,num_filter,kernel_size=2,
                                            padding='same',strides=(2,2),
                                            kernel_initializer=tf.random_normal_initializer())

    return output



def separable_conv(input_data, output_c, training, name, downsample=False):
    """
    :param name:
    :param input_data: shape 为NHWC
    :param output_c: channel of output data
    :param training: 是否在训练，必须为tensor
    :param downsample: 是否下采样
    :return: 输出数据的shape为(N, H, W, output_channel)
    """
    with tf.variable_scope(name):
        input_c = input_data.get_shape().as_list()[-1]  # channel of input_data
        with tf.variable_scope('depthwise'):
            if downsample:
                pad_data = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
                input_data = tf.pad(input_data, pad_data, 'CONSTANT')
                strides = (1, 2, 2, 1)
                padding = 'VALID'
            else:
                strides = (1, 1, 1, 1)
                padding = "SAME"

            dwise_weight = tf.get_variable(name='depthwise_weights', dtype=tf.float32, trainable=True,
                                           shape=(3, 3, input_c, 1),
                                           initializer=tf.random_normal_initializer(stddev=0.01))
            dwise_conv = tf.nn.depthwise_conv2d(input=input_data, filter=dwise_weight, strides=strides, padding=padding)
            dwise_conv = batch_normalization(input_data=dwise_conv, training=training)
            dwise_conv = tf.nn.relu6(dwise_conv)

        with tf.variable_scope('pointwise'):
            pwise_weight = tf.get_variable(name='pointwise_weights', dtype=tf.float32, trainable=True,
                                           shape=(1, 1, input_c, output_c),
                                           initializer=tf.random_normal_initializer(stddev=0.01))
            pwise_conv = tf.nn.conv2d(input=dwise_conv, filter=pwise_weight, strides=(1, 1, 1, 1), padding="SAME")
            pwise_conv = batch_normalization(input_data=pwise_conv, training=training)
            pwise_conv = tf.nn.relu6(pwise_conv)
        return pwise_conv


def inverted_residual(name, input_data, input_c, output_c, training, downsample=False, t=6):
    """
    倒残差结构
    :param name:
    :param input_data: shape 为NHWC
    :param input_c: channel of input data
    :param output_c: channel of output data
    :param training: 是否在训练，必须为tensor
    :param downsample: 是否下采样
    :param t: expansion factor
    :return: 输出数据的shape为(N, H, W, output_channel)
    """
    with tf.variable_scope(name):
        expand_c = t * input_c

        with tf.variable_scope('expand'):#升维
            if t > 1:
                expand_weight = tf.get_variable(name='weights', dtype=tf.float32, trainable=True,
                                                shape=(1, 1, input_data.get_shape().as_list()[3], expand_c),
                                                initializer=tf.random_normal_initializer(stddev=0.01))
                expand_conv = tf.nn.conv2d(input=input_data, filter=expand_weight, strides=(1, 1, 1, 1), padding="SAME")#1X1卷积进行升维操作
                expand_conv = batch_normalization(input_data=expand_conv, training=training)
                expand_conv = tf.nn.relu6(expand_conv)
            else:
                expand_conv = input_data

        with tf.variable_scope('depthwise'):
            if downsample:
                pad_data = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
                expand_conv = tf.pad(expand_conv, pad_data, 'CONSTANT')
                strides = (1, 2, 2, 1)
                padding = 'VALID'
            else:
                strides = (1, 1, 1, 1)
                padding = "SAME"
            dwise_weight = tf.get_variable(name='depthwise_weights', dtype=tf.float32, trainable=True,
                                           shape=(3, 3, expand_conv.get_shape().as_list()[3], 1),
                                           initializer=tf.random_normal_initializer(stddev=0.01))
            dwise_conv = tf.nn.depthwise_conv2d(input=expand_conv, filter=dwise_weight, strides=strides, padding=padding)
            dwise_conv = batch_normalization(input_data=dwise_conv, training=training)
            dwise_conv = tf.nn.relu6(dwise_conv)

        with tf.variable_scope('project'):
            pwise_weight = tf.get_variable(name='weights', dtype=tf.float32, trainable=True,
                                           shape=(1, 1, dwise_conv.get_shape().as_list()[3], output_c),
                                           initializer=tf.random_normal_initializer(stddev=0.01))
            pwise_conv = tf.nn.conv2d(input=dwise_conv, filter=pwise_weight, strides=(1, 1, 1, 1), padding="SAME")
            pwise_conv = batch_normalization(input_data=pwise_conv, training=training)
        if downsample or pwise_conv.get_shape().as_list()[3] != input_data.get_shape().as_list()[3]:
            return pwise_conv
        else:
            return input_data + pwise_conv