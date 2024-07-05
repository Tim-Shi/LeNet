#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2024/7/4 23:10
# @Author: Tingyu Shi
# @File: train.py
# @Description: 训练LeNet模型
import argparse
import copy
import time

import torch
from torch import optim, nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt
import pandas as pd
from model import LeNet


def train_val_dataloader(batches):
    """
    下载并加载数据集
    :return: train_loader, val_loader
    """
    dataset = FashionMNIST(root='./dataset',
                           train=True,
                           transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(28)]),
                           download=True)
    train_dataset, val_dataset = data.random_split(dataset, [round(0.8 * len(dataset)), round(0.2 * len(dataset))])
    train_dataloader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batches,
                                       shuffle=True,
                                       num_workers=0)
    val_dataloader = data.DataLoader(dataset=val_dataset,
                                     batch_size=batches,
                                     shuffle=True,
                                     num_workers=0)
    return train_dataloader, val_dataloader


def train_model(model, train_loader, val_loader, epochs, batches):
    """
    训练模型
    :param model:
    :param train_loader:
    :param val_loader:
    :param epochs:
    :return: train_process
    """
    # 设置训练所用到的设备，有GPU用GPU没有GPU用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 设置梯度下降法更新参数的优化器，此处使用Adam，学习率为0.001
    optimizer = optim.Adam(model.parameters(), 0.001)
    # 设置损失函数为交叉熵函数
    loss_func = nn.CrossEntropyLoss()
    # 将模型放入到训练设备中
    model = model.to(device)

    # 初始化参数
    # 最高准确率的模型参数，初始值为当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    # 最高准确率，初始值为0
    best_acc = 0.0
    # 训练集平均损失值列表
    train_loss_list = []
    # 验证集平均损失值列表
    val_loss_list = []
    # 训练集准确率列表
    train_acc_list = []
    # 验证集准确率列表
    val_acc_list = []
    # 训练开始时间
    since_time = time.time()
    # 训练耗费时间列表
    elapsed_time_list = []

    # 开始训练
    for epoch in range(epochs):

        # 打印当前epoch
        print('-' * 33)
        print('epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 33)

        # 初始化参数
        # 训练集损失值
        train_loss = 0.0
        # 验证集损失值
        val_loss = 0.0
        # 训练集预测正确的数量
        train_corrects = 0
        # 验证集预测正确的数量
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个batch训练和计算
        for step, (images, labels) in enumerate(train_loader):
            # 将图像images和标签labels放入到训练设备中
            images, labels = images.to(device), labels.to(device)
            # 设置模型为训练模式
            model.train()
            # 前向传播计算，输入为一个batch，输出为一个batch中对应的预测
            outputs = model(images)
            # 查找每一行输出中最大值的索引
            predict_label_idx = torch.argmax(outputs, 1)
            # 计算每一个batch的损失值
            loss = loss_func(outputs, labels)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据反向传播的梯度信息来更新网络的参数，以降低loss值的
            optimizer.step()

            # 累加每个batch的损失值
            train_loss += loss.item() * images.size(0)
            # 累加每个batch中预测正确的数量，如果预测正确，则train_correct+1
            train_corrects += torch.sum(predict_label_idx == labels).item()
            # 累加用于训练的每个batch的样本数量
            train_num += images.size(0)

        # 对每一个batch验证和计算
        for step, (images, labels) in enumerate(val_loader):
            # 将图像images和标签labels放入到训练设备中
            images, labels = images.to(device), labels.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播计算，输入为一个batch，输出为一个batch中对应的预测
            outputs = model(images)
            # 查找每一行输出中最大值的索引
            predict_label_idx = torch.argmax(outputs, 1)
            # 计算每一个batch的损失值
            loss = loss_func(outputs, labels)

            # 累加每个batch的损失值
            val_loss += loss.item() * images.size(0)
            # 累加每个batch中预测正确的数量，如果预测正确，则val_correct+1
            val_corrects += torch.sum(predict_label_idx == labels).item()
            # 累加用于验证的每个batch的样本数量
            val_num += images.size(0)

        # 计算并保存每epoch的loss和acc
        train_loss_list.append(train_loss / train_num)
        val_loss_list.append(val_loss / val_num)
        train_acc_list.append(train_corrects / train_num)
        val_acc_list.append(val_corrects / val_num)

        # 打印每epoch的最终的loss和acc
        print('{} train_loss:{:.4f} train_acc: {:.4f}'.format(epoch + 1, train_loss_list[-1], train_acc_list[-1]))
        print('{} val_loss:{:.4f} val_acc: {:.4f}'.format(epoch + 1, val_loss_list[-1], val_acc_list[-1]))

        if val_acc_list[-1] > best_acc:
            # 更新最高准确度
            best_acc = val_acc_list[-1]
            # 更新最高准确度的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时
        elapsed_time = time.time() - since_time
        elapsed_time = '{:.0f}m{:.0f}s'.format(elapsed_time // 60, elapsed_time % 60)
        print('训练和验证耗费的时间:', elapsed_time)
        elapsed_time_list.append(elapsed_time)

    # 保存model、loss和acc
    torch.save(best_model_wts, './best_model_epochs' + str(epochs) + '_batches' + str(batches) + '.pth')
    train_process = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'train_loss_list': train_loss_list,
        'train_acc_list': train_acc_list,
        'val_loss_list': val_loss_list,
        'val_acc_list': val_acc_list,
        'time_elapsed': elapsed_time_list
    })
    train_process.to_csv('loss_acc_epochs' + str(epochs) + '_batches' + str(batches) + '.csv', index=False)
    return train_process


def matplot_loss_acc(train_process, epochs, batches):
    """
    显示每一次迭代后的训练集和验证集的loss和acc
    :param train_process:
    :param epochs:
    :return: None
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_list, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_list, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.subplot(1, 2, 2)
    # plt.ylim(0, 1)
    # plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(train_process['epoch'], train_process.train_acc_list, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_list, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig('./loss_acc_epochs' + str(epochs) + '_batches' + str(batches) + '.png')
    # plt.show()


if __name__ == '__main__':
    """
    训练model
    """
    # 设置命令行参数，输入epochs和batches
    parser = argparse.ArgumentParser(description='命令行参数')
    parser.add_argument('--epochs', '-e', type=int, help='训练轮次，非必须参数,默认值为10', default=10)
    parser.add_argument('--batches', '-b', type=int, help='batch大小，非必须参数,默认值为32', default=32)
    args = vars(parser.parse_args())
    # 打印训练轮次和batch大小
    print('-' * 33)
    print(args)
    epochs = args['epochs']
    batches = args['batches']
    # 实例化模型
    LeNet = LeNet()
    # 加载数据集
    train_dataloader, val_dataloader = train_val_dataloader(batches)
    # 训练模型
    train_process = train_model(LeNet, train_dataloader, val_dataloader, epochs, batches)
    # 根据loss和acc绘制统计图
    matplot_loss_acc(train_process, epochs, batches)
