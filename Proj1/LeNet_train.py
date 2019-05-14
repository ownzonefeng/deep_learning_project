from LeNet import LeNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PairDataset import PairDataset
import dlc_practical_prologue as prologue


data = prologue.generate_pair_sets(1000)
data_train = PairDataset(data, train=True, aux_labels=True)
data_test = PairDataset(data, train=False, aux_labels=True)
data_train_loader = DataLoader(data_train, batch_size=100, shuffle=False)
data_test_loader = DataLoader(data_test, batch_size=100)


net = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-2)


def train(epoch):
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels, digits) in enumerate(data_train_loader):
        images = torch.unsqueeze(images[:, 0], 1)
        labels = digits[:, 0]
        optimizer.zero_grad()
        output = net(images)

        pred = output.detach().max(1)[1]

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i + 1)

        loss.backward()
        optimizer.step()
    print('Train loss:', torch.mean(torch.tensor(loss_list)).item())
    return loss_list, batch_list, epoch


def test(epoch):
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels, digits) in enumerate(data_test_loader):
        images = torch.unsqueeze(images[:, 0], 1)
        labels = digits[:, 0]
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]

        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Epoch: %d, Test Avg. Loss: %f, Accuracy: %f' % (
        epoch, avg_loss.detach().item(), float(total_correct) / len(data_test)))


for i in range(25):
    _, _, epo = train(i)
    test(epo + 1)
