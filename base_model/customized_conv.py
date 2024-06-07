import torch
import torch.nn as nn
import torch.nn.functional as F
class conv(nn.Module):
    def __init__(self, layer_inf,af=nn.ReLU(),pool=nn.MaxPool2d(2,2)) -> None:
        super().__init__()
        self.af=af
        self.pool=pool
        conv_layers=[]
        #self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=3)  #1为in_channels 10为out_channels
        #self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3)
        for i in layer_inf:
            conv_layers.append(nn.Conv2d(*i))
            conv_layers.append(self.pool)
            conv_layers.append(self.af)
        self.conv_layers=nn.Sequential(*conv_layers)
    def forward(self,x):
        #x =self.conv1(x)
        #x = self.conv2(x)
        x=self.conv_layers(x)
        return x



if __name__=="__main__":
    net=conv([(5,3,3,1),(3,3,3,1),(3,1,3,1)],nn.ReLU(),nn.MaxPool2d(2))
    print(net)

