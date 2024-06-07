import sys,os
sys.path.append(os.path.realpath('.'))

import torch
import torch.nn as nn
from base_model.customized_MLP import MLP
from base_model.customized_conv import conv
from train_test.trainer import trainer
from train_test.tester import classifier_test


class CNN(nn.Module):
    def __init__(self,conv_setting,fc_setting) -> None:
        super().__init__()
        self.conv=conv(*conv_setting)
        self.fc=MLP(*fc_setting)
        self.ap =nn.AdaptiveAvgPool1d(fc_setting[0])
    
    def forward(self,x):
        batch_size=x.shape[0]
        x=self.conv(x)
        x=x.view(batch_size,-1)
        x=self.ap(x)
        x=self.fc(x)
        return x

if __name__=="__main__":


    import torch
    from torchvision import transforms
    from torchvision import datasets
    from torch.utils.data import DataLoader
    #数据的准备
    batch_size = 1024
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
    train_loader = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=batch_size
                            )
    test_dataset = datasets.MNIST(root='../dataset/mnist/',
                                train=False,
                                download=True,
                                transform=transform
                                )
    test_loader = DataLoader(test_dataset,
                            shuffle=False,
                            batch_size=batch_size
                            )
    model=CNN([[(1,10,3,1,1),(10,20,3,1,1)],nn.ReLU(),nn.MaxPool2d(2,2)],[320,10,[20,10]])
    print(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainer=trainer(model,train_loader,20,1e-2,device=device,loss=nn.CrossEntropyLoss)
    tester=classifier_test(model,test_loader=test_loader,device=device)
    print(tester)




