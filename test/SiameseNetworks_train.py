import dlc_practical_prologue as prologue
from SiameseNet import SiameseNet
from PairDataset import PairDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader

data = prologue.generate_pair_sets(1000)
data_train = PairDataset(data, train=True, aux_labels=False)
data_test = PairDataset(data, train=False, aux_labels=False)
data_train_loader = DataLoader(data_train, batch_size=10, shuffle=True, num_workers=12)
data_test_loader = DataLoader(data_test, batch_size=40, num_workers=12)

net = SiameseNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1.2e-2)


def train(epoch):
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()

        output = net(images, w_share=False, aux_loss=False)
        labels = labels.float()
        loss_contrastive = torch.mean((1 - labels) * torch.pow(output, 2) + labels * torch.pow(torch.clamp(2 - output, min=0.0), 2))

        loss_list.append(loss_contrastive.detach().item())
        batch_list.append(i + 1)

        loss_contrastive.backward()
        optimizer.step()
    return loss_list, batch_list, epoch


def test(epoch):
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images, w_share=False, aux_loss=False)
        labels = labels.float()
        avg_loss += torch.mean((1 - labels) * torch.pow(output, 2) + labels * torch.pow(torch.clamp(2 - output, min=0.0), 2))
        pred = output.detach()
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Epoch: %d, Test Avg. Loss: %f, Accuracy: %f' % (
        epoch, avg_loss.detach().item(), float(total_correct) / len(data_test)))


for i in range(3):
    loss, batch, epo = train(i)
    test(epo + 1)
