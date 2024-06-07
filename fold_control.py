import torch
from torchvision import transforms
from torchvision import datasets

import os
import torch
print(torch.cuda.is_available())
# 获取当前工作目录
current_working_directory = os.getcwd()

# 打印输出到控制台
print(current_working_directory)

#数据的准备
batch_size = 64
#神经网络希望输入的数值较小，最好在0-1之间，所以需要先将原始图像(0-255的灰度值)转化为图像张量（值为0-1）
#仅有灰度值->单通道   RGB -> 三通道 读入的图像张量一般为W*H*C (宽、高、通道数) 在pytorch中要转化为C*W*H
transform = transforms.Compose([
    #将数据转化为图像张量
    transforms.ToTensor(),
    #进行归一化处理，切换到0-1分布 （均值， 标准差）
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST(root='./dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform
                               )

test_dataset = datasets.MNIST(root='./dataset/mnist/',
                               train=False,
                               download=True,
                               transform=transform
                               )