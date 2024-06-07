import torch
def classifier_test(model,test_loader,device='cpu'):
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            input,label=data
            input,label = input.to(device), label.to(device)
            output=model(input)
            _, predicted = torch.max(output, dim=1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    acc=correct / total
    return acc
