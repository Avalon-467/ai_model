import torch
import torch.nn as nn
import torch.nn.functional as F

def pre_resize(x,block_size):
    shapex=x.shape
    paddingx=(block_size-shapex[-2]%block_size)%block_size
    paddingy=(block_size-shapex[-1]%block_size)%block_size
    if paddingx%2==0:
        a=paddingx//2
        b=paddingx//2
    if paddingx%2==1:
        a=paddingx//2
        b=paddingx//2+1
    if paddingy%2==0:
        c=paddingy//2
        d=paddingy//2
    if paddingy%2==1:
        c=paddingy//2
        d=paddingy//2+1
    x=F.pad(x,[c,d,a,b])
    return x

def atpad(x,y):
    padding=y.shape[-1]-x.shape[-1]
    a=0
    b=0
    if padding!=0:
        if padding%2==0:
            a=padding//2
            b=padding//2
        if padding%2==1:
            a=padding//2
            b=padding//2+1
        x=F.pad(x,[a,b,a,b])
    return x
    
class reset_layer(nn.Module):
    def __init__(self,block_size=10):
        super(reset_layer,self).__init__()
        self.block_size=block_size
        self.leftm=nn.Parameter(torch.eye(int(self.block_size),dtype=float))
        #self.leftm=nn.Parameter(torch.tensor([[0,1],[1,0]],dtype=float))
        self.rightm=nn.Parameter(torch.eye(int(self.block_size),dtype=float))
        #self.rightm=nn.Parameter(torch.tensor([[0,1],[1,0]],dtype=float))
    
    def forward(self,x):
        
        shapex=x.shape
        paddingx=(self.block_size-shapex[-2]%self.block_size)%self.block_size
        paddingy=(self.block_size-shapex[-1]%self.block_size)%self.block_size
        a=0
        b=0
        c=0
        d=0
        if paddingx!=0 or paddingy!=0:
            if paddingx%2==0:
                a=paddingx//2
                b=paddingx//2
            if paddingx%2==1:
                a=paddingx//2
                b=paddingx//2+1
            if paddingy%2==0:
                c=paddingy//2
                d=paddingy//2
            if paddingy%2==1:
                c=paddingy//2
                d=paddingy//2+1
            x=F.pad(x,[c,d,a,b])
        
        identity=x
        list1=torch.chunk(x,self.block_size,-1)
        templist=[]
        for l in self.leftm:
            temp=0
            for i in range(self.block_size):
                temp=temp+l[i].item()*list1[i]
            templist.append(temp)
        x=torch.cat(templist,-1)

        list2=torch.chunk(x,self.block_size,-2)
        templist=[]   

        rightm2=self.rightm.transpose(0,1)
        for r in  rightm2:
            temp=0
            for i in range(self.block_size):
                temp=temp+r[i].item()*list2[i]
            templist.append(temp)
        x=torch.cat(templist,-2)
        x=x+identity
        return x 



