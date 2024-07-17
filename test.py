#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2024/7/5 20:06
# @Author: Tingyu Shi
# @File: test.py
# @Description: 测试模型


import csv
import argparse
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet


def test_dataloader(batches):
    """
    下载并加载数据集
    :param batches:
    :return: test_dataloader, class_label
    """
    test_dataset = FashionMNIST(root='./dataset',
                                train=True,
                                transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor()]),
                                download=True)
    test_dataloader = data.DataLoader(dataset=test_dataset,
                                      batch_size=batches,
                                      shuffle=True,
                                      num_workers=0)
    # 测试集分类标签
    class_label = test_dataset.classes
    return test_dataloader, class_label


def test_model(model, best_model_wts, test_dataloader, class_label, batches):
    """
    测试模型
    :param model:
    :param best_model_wts:
    :param test_dataloader:
    :param class_label:
    :param batches:
    :return: None
    """
    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    # 载入训练好的模型参数
    model.load_state_dict(torch.load(best_model_wts))
    # 将模型放入到训练设备中
    model = model.to(device)

    # 初始化参数
    # 测试集预测正确的数量
    test_corrects = 0.0
    # 测试集样本数量
    test_num = 0

    # 设置模型为评估模式
    model.eval()
    # 开始测试，将梯度置为0，只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for images, labels in test_dataloader:
            # print('images.shape', images.shape)
            # print('labels.shape', labels.shape)
            # 将图像images和标签labels放入到训练设备中
            images, labels = images.to(device), labels.to(device)
            # 前向传播计算，输入为一个batch的数据集，输出为一个batch的数据集中对应的预测
            output = model(images)
            # 查找每一行输出中最大值的索引
            predict_label_idx = torch.argmax(output, 1)
            # 累加每个batch中预测正确的数量，如果预测正确，则test_corrects+1
            test_corrects += torch.sum(predict_label_idx == labels.data).item()
            # 累加用于训练的每个batch的样本数量
            test_num += images.size(0)

            if batches == 1:
                # 查找预测值
                predict_label = predict_label_idx.item()
                # 查找真实值
                label = labels.item()
                print('预测值:', predict_label, class_label[predict_label], '-' * 10, '真实值:', label, class_label[label])

    # 计算测试准确率acc
    test_acc = test_corrects / test_num
    print('=' * 66)
    print("测试的准确率为:", test_acc)
    print('=' * 66)
    # 保存测试准确率acc
    save_dir = './test_result/' + best_model_wts.split('/')[2].split('.')[0] + '.csv'
    with open(save_dir, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['acc'])
        writer.writerow([test_acc])


if __name__ == "__main__":
    """
    测试model
    """
    # 设置命令行参数，输入pth和batches
    # 注意：测试时设置较大的batch_size，可加快测试速度
    parser = argparse.ArgumentParser(description='命令行参数')
    parser.add_argument('--pth', '-p', type=str, help='模型路径，非必须参数，默认值为./train_pth/best_model_epochs50_batches32.pth',
                        default='./train_pth/best_model_epochs50_batches32.pth')
    parser.add_argument('--batches', '-b', type=int, help='batch大小，非必须参数，默认值为1', default=1)
    args = vars(parser.parse_args())
    # 打印模型路径和batch大小
    print('=' * 66)
    print(args)
    best_model_wts = args['pth']
    batches = args['batches']
    # 实例化模型
    model = LeNet()
    # 加载测试数据集
    test_dataloader, class_label = test_dataloader(batches)
    # 测试模型
    test_model(model, best_model_wts, test_dataloader, class_label, batches)
