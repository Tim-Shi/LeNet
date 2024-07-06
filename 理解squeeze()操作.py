#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2024/7/4 23:02
# @Author: Tingyu Shi
# @File: 理解squeeze()操作.py
# @Description: 理解squeeze()操作


import torch


if __name__ == '__main__':
    """
    理解squeeze()操作
    """
    # squeeze()删除维度大小==1的维度
    x = torch.rand(1, 3, 4)
    y = x.squeeze()
    print(x.shape)  # 输出torch.Size([1, 3, 4])
    print(x)
    print(y.shape)  # 输出torch.Size([3, 4])
    print(y)
    print('=' * 50)
    # 注意：squeeze()仅可删除维度大小==1的维度，对其他维度无效
    a = torch.rand(2, 3, 4)
    b = x.squeeze()
    print(a.shape)  # 输出torch.Size([2, 3, 4])
    print(b)
    print(a.shape)  # 输出torch.Size([2, 3, 4])
    print(b)
