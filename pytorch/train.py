#!/usr/bin/env python3
# coding=utf-8

# ============================#
# Program:train.py
#       训练模型
# Date:20-4-16
# Author:liheng
# Version:V1.0
# ============================#

import Model
import m2nistDataSet
import utils_torch
import argparse
import numpy as np
import os
import shutil
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter


class Train(object):
    def __init__(self, config_file: str):
        # 读取配置
        self.config = utils_torch.get_config(config_file)

        # 加载数据
        self.train_dataset = m2nistDataSet.m2nistDataLoader(config_file, 'train')
        self.val_dataset = m2nistDataSet.m2nistDataLoader(config_file, 'val')

        # 创建文件夹
        os.makedirs(self.config['Train.model_save_dir'], exist_ok=True)
        if os.path.exists(self.config['Train.log_dir']):
            shutil.rmtree(self.config['Train.log_dir'])
        os.makedirs(self.config['Train.log_dir'])

        # 加载模型
        self.device = torch.device('cuda'
                                   if (torch.cuda.is_available() and self.config['USE_CUDA'])
                                   else 'cpu')
        self.model = Model.Model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['Train.lr_init'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.99)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.global_setp = 0

        # summary
        self.summary_writer = SummaryWriter(self.config['Train.log_dir'])
        self.summary_writer.add_graph(self.model, (torch.rand([1, 1, 64, 96]),))  # grapth

    def train(self):
        # checkpoint = torch.load(path)
        # self.model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # start_epoch = checkpoint['epoch'] + 1
        try:
            last_model = utils_torch.find_new_file(self.config['Train.model_save_dir'])
            self.model.load_state_dict(torch.load(last_model, map_location=self.device))
            print('[info] Restoring weights from last trained file ...')
        except Exception as e:
            print('[info] Can not find last trained file !!!')
            print('[info] Now it starts to train model from scratch ...')

        class_weights = 10 * [1.0] + [0.2]  # lable为10的class权重为0.2,0-9个class为1，输出一个list
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        for epoch in range(1, 1 + self.config['Train.max_epochs']):
            train_losses, val_losses = [], []
            pbar = tqdm(self.train_dataset)
            for batch in pbar:
                batch_x, batch_y = batch[0].to(self.device), batch[1].to(self.device)
                out = self.model(batch_x)

                loss = Model.CrossEntropyLoss2d(class_weights)(out, batch_y.long())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.global_setp += 1

                train_losses.append(loss.item())

                pbar.set_description("Epoch:%d Step:%d loss:%.3f" % (epoch, self.global_setp, loss.item()))

                # tensorboardX
                self.summary_writer.add_scalar('learning rate', self.optimizer.state_dict()['param_groups'][0]['lr'],
                                               self.global_setp)
                self.summary_writer.add_scalar('train loss', loss, self.global_setp)
                self.summary_writer.add_images('train input images', batch_x, self.global_setp)
                self.summary_writer.add_images('train gt images',
                                               utils_torch.tran_masks2images(batch_y.numpy()),
                                               self.global_setp)
                self.summary_writer.add_images('train pred images',
                                               utils_torch.tran_masks2images(
                                                   torch.argmax(torch.softmax(out, dim=1), dim=1).numpy()),
                                               self.global_setp)

            # 在预测前需要把model设置为评估模式
            self.model.eval()
            with torch.no_grad():  # 无需计算梯度
                for batch_x, batch_y in self.val_dataset:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    out = self.model(batch_x)
                    loss = Model.CrossEntropyLoss2d(class_weights)(out, batch_y.long())

                    val_losses.append(loss.item())

            train_avg_loss, val_avg_loss = np.mean(train_losses), np.mean(val_losses)
            print('epoch:%d, train loss:%.5f, val loss:%.5f ' % (epoch, train_avg_loss, val_avg_loss))

            save_name = os.path.join(self.config['Train.model_save_dir'], 'm2nist-seg_epoch{:d}.pth'.format(epoch))
            # torch.save(self.model.cpu().state_dict(), save_name)#保存cpu的参数
            torch.save(self.model.state_dict(), save_name)

        self.summary_writer.close()


def init_args():
    """
epoch
    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg_pth', type=str,
                        help='The config file path',
                        default='/home/liheng/PycharmProjects/m2nist-segmentation/pytorch/config.yaml')

    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    assert os.path.isfile(args.cfg_pth), args.cfg_pth + 'does not exist !'
    trainer = Train(args.cfg_pth)
    trainer.train()
