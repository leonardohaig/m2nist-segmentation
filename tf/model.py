#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:model.py
#       
#Date:20-4-11
#Author:liheng
#Version:V1.0
#============================#

import common
import evaluator
import tensorflow as tf

class Model(object):
    def __init__(self, input_data,trainable):
        """

        :param input_data: [B,H,W,C] 输入尺寸[B,64,84,1],float32
        :param trainable:
        """
        self.trainable = trainable
        self.decode_ret = self.__build_network(input_data)

    def get_inference(self):
        decode_ret_prob = tf.nn.softmax(self.decode_ret) # [B,H,W,cls]
        # infer_ret = tf.argmax(decode_ret_prob, axis=-1)  # [B,H,W],返回 C 通道最大值的索引
        return decode_ret_prob

    def __build_network(self,input_data):
        """

        :param input_data:
        :return:
        """

        conv = common.convolutional(input_data,filters_shape=(3,3,1,32),
                                    trainable=self.trainable,name='conv1',
                                    downsample=True,activate=True,bn=True)#第一次下采样 [B,32,42,32]
        conv = common.inverted_residual(input_data=conv,input_c=32,
                                        output_c=32,training=self.trainable,
                                        name='expanded_conv2',downsample=False,t=1)#[32,42,32]
        conv1 = conv

        conv = common.inverted_residual(input_data=conv, input_c=32,
                                        output_c=64, training=self.trainable,
                                        name='expanded_conv3', downsample=True)  # 第2次下采样 [16,21,64]
        conv = common.inverted_residual(input_data=conv, input_c=64,
                                        output_c=64, training=self.trainable,
                                        name='expanded_conv4', downsample=False)  # [16,21,64]
        conv2 = conv


        conv = common.inverted_residual(input_data=conv, input_c=64,
                                        output_c=128, training=self.trainable,
                                        name='expanded_conv5', downsample=True)  # 第3次下采样[8,11,128]
        conv = common.inverted_residual(input_data=conv, input_c=128,
                                        output_c=128, training=self.trainable,
                                        name='expanded_conv6', downsample=False)  # [B,8,11,128]
        conv3 = conv

        #================#
        conv = common.convolutional(conv3,[1,1,128,96],trainable=self.trainable,name='decode_conv1')

        conv = tf.image.resize_images(conv,[16,21])#[B,16,21,96]
        conv = common.convolutional(conv,[1,1,96,96],trainable=self.trainable,name='decode_conv2')

        conv = common.route(name='route1',previous_output=conv,current_output=conv2) #[B,16,21,160]
        conv = common.convolutional(conv,[1,1,160,128],trainable=self.trainable,name='decode_conv3')
        conv = common.separable_conv(input_data=conv,output_c=64,training=self.trainable,name='decode_conv4') #[B,16,21,64]

        conv = common.upsample(input_data=conv,name='up',method='deconv')#[B,32,42,64]
        conv = common.convolutional(conv, [1, 1, 64, 64], trainable=self.trainable, name='decode_conv5')

        conv = common.route(name='route2', previous_output=conv, current_output=conv1) # [B,32,41,96]
        conv = common.convolutional(conv, [1, 1, 96, 64], trainable=self.trainable, name='decode_conv6')
        conv = common.separable_conv(input_data=conv, output_c=32, training=self.trainable,name='decode_conv7')  # [B,32,41,32]

        conv = tf.image.resize_images(conv, [64, 84])  # [B,64,84,32]
        conv = common.convolutional(conv, [1, 1, 32, 32], trainable=self.trainable, name='decode_conv8')

        conv = common.separable_conv(input_data=conv,output_c=16,training=self.trainable,name='decode_conv9')#[,,,16]

        conv = common.separable_conv(input_data=conv, output_c=11, training=self.trainable,
                                     name='decode_conv10')  # [,64,84,11]


        return conv

    def computer_loss(self,gt_masks):
        """

        :param gt_masks:
        :return:
        """
        decode_logits_reshape = tf.reshape(
            self.decode_ret,
            shape=[self.decode_ret.get_shape().as_list()[0],
                   -1,
                   self.decode_ret.get_shape().as_list()[3]])

        gt_masks_reshape = tf.reshape(
            gt_masks,
            shape=[gt_masks.get_shape().as_list()[0],
                   -1])
        gt_masks_reshape = tf.one_hot(gt_masks_reshape,depth=11)

        class_weights = 10*[1.0] + [0.2] # lable为10的class权重为0.2,0-9个class为1，输出一个list
        class_weights = tf.convert_to_tensor(class_weights) # 将list转换为tensor,shape为[11,]
        weights_loss = tf.reduce_sum(
            tf.multiply(gt_masks_reshape, class_weights),# 自动进行broadcast,[batch_size,11],[11,],乘积为[B,11]
            -1)
        # logits是神经网络的输出, 注意要求是softmax处理之前的logits,
        # 因为tf.losses.softmax_cross_entropy()方法内部会对logits做softmax处理
        binary_segmentation_loss = tf.losses.softmax_cross_entropy(onehot_labels=gt_masks_reshape,
                                                                   logits=decode_logits_reshape,
                                                                   weights=weights_loss)
        binary_segmentation_loss = tf.reduce_mean(binary_segmentation_loss)

        return binary_segmentation_loss

    # def computer_accuracy(self,gt_masks):
    #     pass

    def get_pred_image_summary(self):

        pass




if __name__ == '__main__':
    graph = tf.get_default_graph()

    input_data = tf.placeholder(dtype=tf.float32, shape=[2, 64, 84, 1], name='input_data')
    # input_data = tf.truncated_normal(shape=(2, 64, 84, 1), dtype=tf.float32)
    trainable = tf.placeholder(dtype=tf.bool, shape=[], name='training')

    model = Model(input_data,tf.constant(False, dtype=tf.bool))

    flops = evaluator.evaluate_flops(graph)
    params = evaluator.evaluate_params(graph)
    print('flops/M:', flops/1e6)
    print('params/M:', params/1e6)