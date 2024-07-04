#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2024/7/4 22:56
# @Author: Tingyu Shi
# @File: 下载并可视化展示数据集.py
# @Description: 下载并可视化展示数据集


from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np


def loader_dataset():
    """
    下载并加载数据集
    :return: train_dataset, train_dataloader
    """
    train_dataset = FashionMNIST(root='./dataset',
                                 train=True,
                                 transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(28)]),
                                 download=True)

    train_dataloader = data.DataLoader(dataset=train_dataset,
                                   batch_size=64,
                                   shuffle=True,
                                   num_workers=4)

    return train_dataset, train_dataloader


def matplot_dataset(x, y):
    """
    可视化数据集
    :param x: 图像images数组
    :param y: 标签labels数组
    :return: None
    """
    plt.figure(figsize=(12, 5))
    for i in np.arange(len(y)):
        plt.subplot(4, 16, i + 1)
        plt.imshow(x[i, :, :], cmap='gray')
        plt.title(class_label[labels[i]], size=10)
        plt.axis("off")
        plt.subplots_adjust(wspace=0.05)
    plt.savefig('./dataset_example.png')
    # plt.show()


if __name__ == '__main__':
    """
    下载并可视化展示数据集
    """
    # 下载并加载数据集
    train_dataset, train_dataloader = loader_dataset()
    # 测试——获取一个batch的数据，并可视化
    for step, (images, labels) in enumerate(train_dataloader):
        # 获取数据
        if step > 0:
            break
        batch_images = images.squeeze().numpy()  # squeeze()删除维度大小为1的维度（灰度图通道维度为1），4维转3维并转换成Numpy数组（图片数据）images
        batch_labels = labels.numpy()  # 1维转换成Numpy数组（标签数据）labels
        class_label = train_dataset.classes  # 训练集的标签
        print('Labels:', class_label)
        print('The size of batch images in train data:', batch_images.shape)  # 每个batch的图像的维度是64*28*28
        print('The size of batch channels in train data:', batch_labels.shape)  # 每个batch的标签的维度是64
        print(batch_labels)
        # 可视化
        matplot_dataset(batch_images, batch_labels)