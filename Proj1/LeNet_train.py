from LeNet import LeNet
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((14, 14)),
                       transforms.ToTensor()]))
data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((14, 14)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(
    data_train, batch_size=200, shuffle=True, num_workers=12)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=12)

'''
data = prologue.generate_pair_sets(1000)
data_train = PairDataset(data, train=True, aux_labels=True)
data_test = PairDataset(data, train=False, aux_labels=True)
data_train_loader = DataLoader(data_train, batch_size=20, shuffle=False)
data_test_loader = DataLoader(data_test, batch_size=20)
'''

net = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1.2e-2)


def train(epoch):
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
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
    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]

        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Epoch: %d, Test Avg. Loss: %f, Accuracy: %f' % (
        epoch, avg_loss.detach().item(), float(total_correct) / len(data_test)))


for i in range(5):
    _, _, epo = train(i)
    test(epo + 1)
