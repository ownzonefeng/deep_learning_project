import torch
from torch import nn
from data import *

train_loader = dataloader(10)
test_loader = dataloader(10)

model = nn.Sequential(nn.Linear(2, 25),
                      nn.ReLU(),
                      nn.Linear(25, 25),
                      nn.ReLU(),
                      nn.Linear(25, 25),
                      nn.ReLU(),
                      nn.Linear(25, 2),
                      nn.Tanh())
optimizer = torch.optim.SGD(model.parameters(),lr=1e-1)

for epoch in range(50):
    for x,label_1d,label_2d in train_loader:
        pred = model(x)
        loss = ((pred-label_2d)**2).mean()
        model.zero_grad()
        loss.backward()
        optimizer.step()
    acc_train = 0
    for x, label_1d, label_2d in train_loader:
        pred_2d = model(x)
        pred_1d = label_2dto1d(pred_2d)
        acc_train += (pred_1d == label_1d).float().sum()
    acc_test = 0
    for x,label_1d,label_2d in test_loader:
        pred_2d = model(x)
        pred_1d = label_2dto1d(pred_2d)
        acc_test += (pred_1d==label_1d).float().sum()
    print('epoch: %d/50 loss: %f train error %2.2f%% test error %2.2f%%'%(epoch, loss, (1-acc_train.item()/1000)*100, (1-acc_test.item()/1000)*100))
