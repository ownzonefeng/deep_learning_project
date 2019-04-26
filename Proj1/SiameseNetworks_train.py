import dlc_practical_prologue as prologue
from SiameseNet import SiameseNet
from PairDataset import PairDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import time

data_train, data_test, data_train_loader, data_test_loader = [None] * 4
weight_sharing_status, aux_labels_status, print_train_data, print_test_data = [None] * 4
net, criterion, optimizer = [None] * 3


def train(epoch):
    net.train()
    total_loss = 0
    total_correct = 0
    for i, (images, labels, digits) in enumerate(data_train_loader):
        optimizer.zero_grad()
        output = net(images, w_share=weight_sharing_status, aux_loss=aux_labels_status)
        if aux_labels_status:
            one_hot_labels_digits_0 = convert_one_hot(digits[:, 0])
            one_hot_labels_digits_1 = convert_one_hot(digits[:, 1])
            one_hot_labels_bool = convert_one_hot(labels)
            loss = criterion(output[0][:, 0:10], one_hot_labels_digits_0)
            loss += criterion(output[0][:, 10:], one_hot_labels_digits_1)
            loss += criterion(output[1], one_hot_labels_bool)
            pred = output[1].detach().max(1)[1]
        else:
            one_hot_labels = convert_one_hot(labels)
            loss = criterion(output, one_hot_labels)
            pred = output.detach().max(1)[1]

        total_loss += loss.detach().item()
        total_correct += pred.eq(labels.view_as(pred)).sum()

        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(data_train)
    accuracy = float(total_correct) / len(data_train)
    if print_train_data:
        print('Epoch: %d, Train Avg. Loss: %f, Accuracy: %f' % (epoch, avg_loss, accuracy))
    return avg_loss, accuracy


def test(epoch):
    net.eval()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels, digits) in enumerate(data_test_loader):
        output = net(images, w_share=weight_sharing_status, aux_loss=aux_labels_status)

        if aux_labels_status:
            one_hot_labels_digits_0 = convert_one_hot(digits[:, 0])
            one_hot_labels_digits_1 = convert_one_hot(digits[:, 1])
            one_hot_labels_bool = convert_one_hot(labels)
            loss_1 = criterion(output[0][:, 0:10], one_hot_labels_digits_0)
            loss_1 += criterion(output[0][:, 10:], one_hot_labels_digits_1)
            loss_1 += criterion(output[1], one_hot_labels_bool)
            pred = output[1].detach().max(1)[1]
        else:
            one_hot_labels = convert_one_hot(labels)
            loss_1 = criterion(output, one_hot_labels)
            pred = output.detach().max(1)[1]

        total_loss += loss_1
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss = float(total_loss) / len(data_test)
    accuracy = float(total_correct) / len(data_test)
    if print_test_data:
        print('Epoch: %d, Test  Avg. Loss: %f, Accuracy: %f' % (epoch, avg_loss, accuracy))
    if print_train_data and print_test_data:
        print('\n')
    return avg_loss, accuracy


def convert_one_hot(original_labels):
    bits = len(torch.unique(original_labels))
    if bits > 2:
        bits = 10
    new_labels = torch.zeros((len(original_labels), bits))
    original_labels = original_labels.view(-1, 1)
    new_labels = new_labels.scatter(1, original_labels, 1)
    return new_labels


def weight_init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        module.reset_parameters()


def data_generation():
    global data_train, data_test, data_train_loader, data_test_loader
    data = prologue.generate_pair_sets(1000)
    data_train = PairDataset(data, train=True, aux_labels=True)
    data_test = PairDataset(data, train=False, aux_labels=True)
    data_train_loader = DataLoader(data_train, batch_size=100, shuffle=True, num_workers=12)
    data_test_loader = DataLoader(data_test, batch_size=100, num_workers=12)


def start_learning(epoch=25, w_sharing=1, aux_labels=1, print_train=1, print_test=1):
    global weight_sharing_status, aux_labels_status, print_train_data, print_test_data
    weight_sharing_status = w_sharing
    aux_labels_status = aux_labels
    print_train_data = print_train
    print_test_data = print_test
    # print("=" * 100)
    # print('Weight sharing:', bool(weight_sharing_status), '   Auxiliary losses:', bool(aux_labels_status))

    global net, criterion, optimizer
    net = SiameseNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adagrad(net.parameters(), lr=1e-2)

    data_generation()
    net.apply(weight_init)

    # para_size = 0
    # for i in net.parameters():
    #     para_size += torch.prod(torch.tensor(i.size()))
    # print('Parameters quantity:', para_size.item() // 2, '\n')

    train_accuracy_list = torch.empty((1, epoch))
    train_loss_list = torch.empty((1, epoch))
    test_accuracy_list = torch.empty((1, epoch))
    test_loss_list = torch.empty((1, epoch))
    train_time_list = torch.empty((1, epoch))

    for i in range(epoch):
        s = time.perf_counter()
        train_loss, train_accuracy = train(i + 1)
        elapse = time.perf_counter() - s
        test_loss, test_accuracy = test(i + 1)

        train_loss_list[0, i] = train_loss
        train_accuracy_list[0, i] = train_accuracy
        test_loss_list[0, i] = test_loss
        test_accuracy_list[0, i] = test_accuracy
        train_time_list[0, i] = elapse
    # print("After %d epoch of training, the accuracy on test set is %f\n" % (epoch, max(test_accuracy_list)))

    net, criterion, optimizer = [None] * 3

    global data_train, data_test, data_train_loader, data_test_loader
    data_train, data_test, data_train_loader, data_test_loader = [None] * 4

    return train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list, train_time_list


if __name__ == '__main__':
    _ = start_learning(epoch=25, w_sharing=0, aux_labels=1, print_train=1, print_test=1)
