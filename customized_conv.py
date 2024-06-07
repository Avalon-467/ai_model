import torch
import torch.nn as nn

class conv(nn.Module):
    def __init__(self, layer_inf,af) -> None:
        super().__init__()
        self.af=af()
        conv_layers=[]
        for i in layer_inf:
            conv_layers.append(nn.Conv2d(*i))
            conv_layers.append(self.af)
        self.conv_layers=nn.Sequential(*conv_layers)
    def forward(self,x):
        x=self.conv_layers(x)
        return x


if __name__=="__main__":
    net=conv([(5,3,3,1),(3,3,3,1),(3,1,3,1)],nn.ReLU)
    print(net)

