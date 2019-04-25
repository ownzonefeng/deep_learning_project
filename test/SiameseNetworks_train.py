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
data_train_loader = DataLoader(data_train, batch_size=100, shuffle=True, num_workers=12)
data_test_loader = DataLoader(data_test, batch_size=100, num_workers=12)

net = SiameseNet()
criterion = nn.BCELoss()
optimizer = optim.Adagrad(net.parameters(), lr=1e-2)

weight_sharing_status = 0
aux_labels_status = 0
print_train_data = 1
print_test_data = 1


def train(epoch):
    net.train()
    loss_list, batch_list = [], []
    total_correct = 0
    for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()
        output = net(images, w_share=weight_sharing_status, aux_loss=aux_labels_status)
        one_hot_labels = convert_one_hot(labels)
        loss = criterion(output, one_hot_labels)
        pred = output.detach().max(1)[1]
        loss_list.append(loss.detach().item())
        batch_list.append(i + 1)

        total_correct += pred.eq(labels.view_as(pred)).sum()

        loss.backward()
        optimizer.step()
    if print_train_data:
        print('Epoch: %d, Train Avg. Loss: %f, Accuracy: %f' % (
        epoch, torch.mean(torch.tensor(loss_list)).item(), float(total_correct) / len(data_train)))
    return loss_list, batch_list, epoch


def test(epoch):
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images, w_share=weight_sharing_status, aux_loss=aux_labels_status)
        one_hot_labels = convert_one_hot(labels)
        loss_1 = criterion(output, one_hot_labels)
        avg_loss += loss_1
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    if print_test_data:
        print('Epoch: %d, Test  Avg. Loss: %f, Accuracy: %f' % (
        epoch, avg_loss.detach().item(), float(total_correct) / len(data_test)))
    if print_train_data and print_test_data:
        print('\n')


def convert_one_hot(original_labels):
    new_labels = torch.zeros((len(original_labels), 2))
    original_labels = original_labels.view(-1, 1)
    new_labels = new_labels.scatter(1, original_labels, 1)
    return new_labels


for i in range(25):
    loss, batch, epo = train(i + 1)
    test(epo)
