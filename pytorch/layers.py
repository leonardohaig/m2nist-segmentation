#!/usr/bin/env python3
# coding=utf-8

# ============================#
# Program:layers.py
#       
# Date:20-4-16
# Author:liheng
# Version:V1.0
# ============================#

import torch.nn as nn
import torch


class InvertedResidual(torch.nn.Module):
    """
    inverted residual block
    """

    def __init__(self, in_channels, out_channels, t=6, s=1):
        """
        Initialization of inverted residual block
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param t: the expansion factor of block
        :param s: stride of conv3X3
        """
        super(InvertedResidual, self).__init__()

        self.s = s

        # conv 1*1
        self.conv1X1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1, bias=False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6())

        # conv 3*3 depthwise
        self.conv3X3 = nn.Sequential(
            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=s, padding=1, groups=in_channels * t, bias=False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6())

        # conv 1*1 linear
        self.conv1X1_Linear = nn.Sequential(
            nn.Conv2d(in_channels * t, out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels))

        # if use conv residual connection
        if in_channels != out_channels and s == 1:
            self.res_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.res_conv = None

    def forward(self, x):
        """

        :param x:
        :return:
        """
        out = self.conv1X1(x)
        out = self.conv3X3(out)
        out = self.conv1X1_Linear(out)

        if self.s == 1:
            # use residual connection
            if self.res_conv is None:
                out = x + out
            else:
                out = self.res_conv(x) + out

        return out


def sepconv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=True):
    """
    conv
    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param bias:
    :return:
    """
    # To preserve the equation of padding. (k=1 maps to pad 0, k=3 maps to pad 1, k=5 maps to pad 2, etc.)
    padding = (kernel_size + 1) // 2 - 1
    return nn.Sequential(
        # DWise
        nn.Conv2d(in_channels, in_channels, 3, stride, padding, bias=bias, groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(),

        # PWise
        nn.Conv2d(in_channels, out_channels, 1, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6())


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=True):
    """
    conv
    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param bias:
    :return:
    """
    # To preserve the equation of padding. (k=1 maps to pad 0, k=3 maps to pad 1, k=5 maps to pad 2, etc.)
    padding = (kernel_size + 1) // 2 - 1
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6())


class EncoderBlock(torch.nn.Module):
    """

    """

    def __init__(self, in_channels, out_channels, t=6):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(sepconv2d(in_channels, out_channels, 3, 2, False),
                                   InvertedResidual(out_channels, out_channels, t=t, s=1))


if __name__ == '__main__':
    input_data = torch.ones([5, 1, 64, 96], dtype=torch.float32)  # [B,C,H,W]

    inver_block = InvertedResidual(1, 16, t=6, s=2)
    conv1 = inver_block(input_data)
    print(conv1.shape)
