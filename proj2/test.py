from modules import *
import torch
from data import *
torch.set_grad_enabled(False)

train_loader = dataloader(10)
test_loader = dataloader(10)

model = Sequential(Linear(2, 25),
                   ReLU(),
                   Linear(25, 25),
                   ReLU(),
                   Linear(25, 25),
                   ReLU(),
                   Linear(25, 2),
                   tanh())
loss_mse = LossMSE(model)
optimizer = SGD(lr=1e-1, model=model)

for epoch in range(50):
    for x,label_1d,label_2d in train_loader:
        pred = model(x)
        loss = loss_mse(pred, label_2d)
        model.zero_grad()
        loss_mse.backward()
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
    print('epoch: %d loss: %f train error %2.2f%% test error %2.2f%%'%(epoch, loss, (1-acc_train.item()/1000)*100, (1-acc_test.item()/1000)*100))