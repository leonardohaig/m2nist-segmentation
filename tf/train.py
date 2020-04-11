#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:train.py
#       
#Date:20-4-11
#Author:liheng
#Version:V1.0
#============================#

import model
import Data_provider
import utils
import os
import tensorflow as tf
from multiprocessing import cpu_count
import glog as log
from tqdm import tqdm
import argparse
import numpy as np
import time
import shutil

class Train(object):
    def __init__(self,config_file:str):
        self.config = utils.get_config(config_file)

        # 构造数据队列
        self.__init_batch_queue()

        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = self.config['Train.tf_allow_growth']
        self.tf_config.gpu_options.per_process_gpu_memory_fraction = self.config['Train.gpu_memory_fraction']  # 占用80%显存
        self.tf_config.gpu_options.allocator_type = 'BFC'
        self.sess = tf.Session(config=self.tf_config)

        #创建文件夹
        os.makedirs(self.config['Train.model_save_dir'],exist_ok=True)
        if os.path.exists(self.config['Train.log_dir']):
            shutil.rmtree(self.config['Train.log_dir'])
        os.makedirs(self.config['Train.log_dir'])

        with tf.name_scope('define_input'):
            self.trainable = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope('define_loss'):
            def true_fn():
                return self.train_data_batch_queue.dequeue()
            def false_fn():
                return self.val_data_batch_queue.dequeue()

            self.img_bath,self.instance_batch = tf.cond(pred=self.trainable,
                                                        true_fn=true_fn,
                                                        false_fn=false_fn)
            self.model = model.Model(self.img_bath,self.trainable)
            self.net_var = tf.global_variables()
            self.total_loss = self.model.computer_loss(self.instance_batch)

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1,trainable=False,name='global_step')

            # 学习率多项式衰减
            self.learn_rate = tf.train.polynomial_decay(learning_rate=self.config['Train.lr_init'],
                                                        global_step=self.global_step,
                                                        decay_steps=500,
                                                        end_learning_rate=self.config['Train.lr_end'],
                                                        power=0.5, cycle=True)

            global_step_update = tf.assign_add(self.global_step, 1)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(
                self.config['Train.moving_ave_decay']).apply(
                tf.trainable_variables())  # 给模型中的变量创建滑动平均（滑动平均，作用于模型中的变量）

        with tf.name_scope("defin_train"):
            trainable_var_list = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss,var_list=trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()
        # with tf.name_scope('load_defined_params'):
        #     """
        #     从模型中恢复指定层的参数到网络中
        #     """
        #     self.net_var = []
        #     for var in tf.trainable_variables():
        #         var_name = var.op.name
        #         if not ('BatchNorm' in var_name):
        #             self.net_var.append(var)
        #         else:
        #             print(var_name)

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)  # 仅保留最近3次的结果

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate", self.learn_rate)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.image("input_gt_image", self.img_bath, max_outputs=3)
            # tf.summary.image("train_pred_image", self.train_pred_image, max_outputs=3)

            self.write_op = tf.summary.merge_all()  # 将所有summary全部保存到磁盘,以便tensorboard显示
            self.summary_writer = tf.summary.FileWriter(
                self.config['Train.log_dir'], graph=self.sess.graph)  # 指定一个文件用来保存图

    def __init_batch_queue(self):
        train_imgs_pth = self.config['Train.images_pth']
        train_masks_pth = self.config['Train.masks_pth']
        val_imgs_pth = self.config['Val.images_pth']
        val_masks_pth = self.config['Val.masks_pth']

        train_dataset = Data_provider.Data_provider(train_imgs_pth, train_masks_pth)
        _data_train = train_dataset.next_batch(batch_size=self.config['Train.batch_size'])
        self.train_data_batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            _data_train,num_threads=cpu_count()//3)
        self.steps_per_trainepoch = train_dataset.imgs_num // self.config['Train.batch_size']

        val_dataset = Data_provider.Data_provider(val_imgs_pth, val_masks_pth)
        _data_val = val_dataset.next_batch(batch_size=self.config['Val.batch_size'])
        self.val_data_batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            _data_val, num_threads=cpu_count()//3)
        self.steps_per_valepoch = val_dataset.imgs_num // self.config['Val.batch_size']


    def train(self):
        self.sess.run(tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()))

        # 加载上次已经训练后的权重
        try:
            log.info('Restoring weights from last trained file ...')
            last_checkpoint = tf.train.latest_checkpoint(self.config['Train.model_save_dir'])  # 会自动找到最近保存的变量文件
            self.loader.restore(self.sess, last_checkpoint)
        except:
            log.warning('Can not find last trained file !!!')
            log.info('Now it starts to train model from scratch ...')

        cood = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=cood)

        for epoch in range(1,1+self.config['Train.max_epochs']):
            pbar = tqdm(range(self.steps_per_trainepoch))
            train_epoch_loss, val_epoch_loss = [], []


            for step in pbar:
                _,summary,train_step_loss,global_step_val = self.sess.run(
                    [self.train_op_with_all_variables,self.write_op,self.total_loss,self.global_step],
                    feed_dict={self.trainable:True} )

                global_step_val = int(global_step_val)
                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("Step:%d train loss: %.2f" % (step+1, train_step_loss))

                # 每500step额外保存一个模型
                if global_step_val % 500 == 0:
                    ckpt_file = os.path.join(
                        self.config['Train.model_save_dir'], 'm2nist_model_%d-epoch.ckpt' % epoch)
                    self.saver.save(self.sess, ckpt_file, global_step=global_step_val)

            for step in range(self.steps_per_valepoch):
                val_step_loss = self.sess.run(self.total_loss,
                                               feed_dict={self.trainable:False})

                val_epoch_loss.append(val_step_loss)

            train_epoch_loss, val_epoch_loss = np.mean(train_epoch_loss), np.mean(val_epoch_loss)

            ckpt_file = os.path.join(self.config['Train.model_save_dir'],
                                     'm2nist_model-val_loss=%.4f_%d-epoch.ckpt' % (val_epoch_loss, epoch))
            self.saver.save(self.sess, ckpt_file, global_step=global_step_val)

            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            log.info("=> Epoch: %2d/%2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                  % (epoch, self.config['Train.max_epochs'], log_time, train_epoch_loss,val_epoch_loss, ckpt_file))


        cood.request_stop()
        cood.join(threads=threads)


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg_pth', type=str,
                        help='The config file path',
                        default='/home/liheng/PycharmProjects/m2nist-segmentation/tf/config.yaml')

    return parser.parse_args()

if __name__ == '__main__':
    args = init_args()

    trainer = Train(args.cfg_pth)
    trainer.train()
