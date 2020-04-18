#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:Model.py
#       
#Date:20-4-16
#Author:liheng
#Version:V1.0
#============================#

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        # encoder
        self.encoder1 = self.EncoderBlock(1,16)
        self.encoder2 = self.EncoderBlock(16,32)
        self.encoder3 = self.EncoderBlock(32,64)
        self.encoder4 = self.EncoderBlock(64,96)

        # decoder
        self.decode4 = self.DecodeBlock(96,64)
        self.decode3 = self.DecodeBlock(128,32)
        self.decode2 = self.DecodeBlock(64,16)
        self.decode1 = self.DecodeBlock(32,16)

        self.res_conv = layers.conv2d(16,11,3,1)

    def EncoderBlock(self,in_channels, out_channels, t=6):
        return torch.nn.Sequential(layers.sepconv2d(in_channels,out_channels,3,2,False),
                                   layers.InvertedResidual(out_channels,out_channels,t=t,s=1))
    def DecodeBlock(self,in_channels,out_channels,kernel_size=3,bias=True):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param bias:
        :return:
        """
        return torch.nn.Sequential(
            # conv1x1
            nn.Conv2d(in_channels,in_channels//4,1,bias=bias),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU6(),

            #deconv 3X3
            nn.ConvTranspose2d(in_channels//4,in_channels//4,kernel_size,stride=2,padding=1,output_padding=1,bias=bias),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU6(),

            # conv1x1
            nn.Conv2d(in_channels//4,out_channels,1,bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6())

    def forward(self, x):
        #encode stage
        e1 = self.encoder1(x) # [B,16,32,48]
        e2 = self.encoder2(e1) # [B,32,16,24]
        e3 = self.encoder3(e2) # [B,64,8,12]
        e4 = self.encoder4(e3) # [B,96,4,6]

        #decode stage
        d4 = torch.cat((self.decode4(e4),e3),dim=1) # [B,64+64,8,12]
        d3 = torch.cat((self.decode3(d4),e2),dim=1) #[B,32+32,16,24]
        d2 = torch.cat((self.decode2(d3),e1),dim=1) #[B,16+16,32,48]
        d1 = self.decode1(d2) #[B,16,64,96]

        #res
        res = self.res_conv(d1) #[B,11,64,96]
        return res


class CrossEntropyLoss2d(nn.Module):
    """
    defines a cross entropy loss for 2D images
    """
    def __init__(self, weight=None, ignore_label= 255):
        """
        :param weight: 1D weight vector to deal with the class-imbalance
        Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network.
        You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.
        """
        super().__init__()

        #self.loss = nn.NLLLoss2d(weight, ignore_index=255)
        # self.loss = nn.NLLLoss(weight)
        self.loss = nn.CrossEntropyLoss(weight)

    def forward(self, outputs, targets):
        # return self.loss(F.log_softmax(outputs, 1), targets)
        return self.loss(outputs,targets)


if __name__ == '__main__':
    from torchstat import stat

    # initial model
    model = Model()

    input_data = torch.ones([5, 1, 64, 96], dtype=torch.float32)  # [B,C,H,W]

    stat(model,(1,64,96))

    exit(0)


    # initialize the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # print the model's state_dict
    print("model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())

    print("\noptimizer's state_dict")
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])