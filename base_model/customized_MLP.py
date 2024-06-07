import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_shape,output_shape,layer_node,af=nn.ReLU()) -> None:
        super().__init__()
        self.input_layer=nn.Linear(input_shape,layer_node[0])
        self.af=af
        hidden_layers=[]
        layer_node
        for i in range(len(layer_node)-1):
            hidden_layers.append(nn.Linear(layer_node[i],layer_node[i+1]))
            hidden_layers.append(self.af)
        self.hidden_layers=nn.Sequential(*hidden_layers)
        self.output_layer=nn.Linear(layer_node[-1],output_shape)
        
    def forward(self,x):
        x=self.input_layer(x)
        x=self.af(x)
        x=self.hidden_layers(x)
        x=self.output_layer(x)
        return x


if __name__=="__main__":
    net=MLP(1,10,[10,3,4,500,10],nn.ReLU())
    print(net(torch.tensor([1],dtype=torch.float)))

