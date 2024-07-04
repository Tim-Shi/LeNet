from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    train_dataset = FashionMNIST(root='./dataset',
                                 train=True,
                                 transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(224)]),
                                 download=True)

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=64,
                                   shuffle=True,
                                   num_workers=4)

    # 测试——获取一个Batch的数据，并可视化
    for step, (b_x, b_y) in enumerate(train_loader):
        # 获取数据
        if step > 0:
            break
        batch_x = b_x.squeeze().numpy()  # 将四维张量移除第1维，并转换成Numpy数组
        batch_y = b_y.numpy()  # 将张量转换成Numpy数组
        class_label = train_dataset.classes  # 训练集的标签
        print('Labels:', class_label)
        print('The size of batch in train data:', batch_x.shape)  # 每个mini-batch的维度是64*224*224
        # 可视化
        plt.figure(figsize=(12, 5))
        for i in np.arange(len(batch_y)):
            plt.subplot(4, 16, i + 1)
            plt.imshow(batch_x[i, :, :], cmap='gray')
            plt.title(class_label[batch_y[i]], size=10)
            plt.axis("off")
            plt.subplots_adjust(wspace=0.05)
        plt.show()

