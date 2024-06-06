import torch
import torch.utils
from torch.utils.data import DataLoader,Dataset
import tqdm
import customized_MLP
def trainer(model,loss,optim,train_loader,epoch,lr,optim_setting):
    optimizer=optim(model.parameters(),lr=lr)
    criterion=loss()
    loss_list=[]
    for i in tqdm.tqdm(range(epoch)):
        epoch_loss=0
        for data in train_loader:
            input,label=data
            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
        loss_list.append(epoch_loss)
    return loss_list


def test(model,test_loader):
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            input,label=data
            output=model(input)
            _, predicted = torch.max(output, dim=1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    acc=correct / total
    return acc


if __name__=="__main__":
    net=customized_MLP.MLP(1,1,[10,10,10],torch.nn.ReLU)
    def sq(x):
        return x**2

    import random

    # 定义采样范围
    start = 0
    end = 1000
    
    train_i=[]
    train_l=[]
    test_i=[]
    test_l=[]
    for i in range(1000):
        # 生成均匀采样的数字
        sampled_number = random.uniform(start, end)
        train_i.append([sampled_number])
        train_l.append([sq(sampled_number)])
    for i in range(100):
        # 生成均匀采样的数字
        sampled_number = random.uniform(start, end)
        test_i.append([sampled_number])
        test_l.append([sq(sampled_number)])
    
    class supervise_dataset(Dataset):
        def __init__(self,train_i,train_l):
            self.train_i=torch.tensor(train_i)
            self.train_l=torch.tensor(train_l)
        
        def __len__(self):
            return self.train_i.size(0)
        
        def __getitem__(self,index):
            return self.train_i[index],self.train_l[index]
        
    train_dataset=supervise_dataset(train_i,train_l)
    test_dataset=supervise_dataset(test_i,test_l)

    batch_size=20

    train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size
                            )
    test_loader = DataLoader(test_dataset,
                            shuffle=False,
                            batch_size=batch_size
                            )

    train_acc=trainer(net,torch.nn.L1Loss, torch.optim.Adam,train_loader,1000,1e-2,None)
    while 1:
        x=input("int")
        print(net(torch.tensor([[int(x)]],dtype=torch.float)))
    



   

            











    
        










