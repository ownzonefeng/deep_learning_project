import dlc_practical_prologue as prologue
from SiameseNet import SiameseNet
from PairDataset import PairDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader

data = prologue.generate_pair_sets(1000)
data_train = PairDataset(data, train=True, aux_labels=True)
data_test = PairDataset(data, train=False, aux_labels=True)
data_train_loader = DataLoader(data_train, batch_size=100, shuffle=True, num_workers=12)
data_test_loader = DataLoader(data_test, batch_size=100, num_workers=12)

net = SiameseNet()
criterion = nn.BCELoss()
optimizer = optim.Adagrad(net.parameters(), lr=1e-2)

weight_sharing_status = 1
aux_labels_status = 1
print_train_data = 1
print_test_data = 1


def train(epoch):
    net.train()
    total_loss = 0
    total_correct = 0
    for i, (images, labels, digits) in enumerate(data_train_loader):
        optimizer.zero_grad()
        output = net(images, w_share=weight_sharing_status, aux_loss=aux_labels_status)
        if aux_labels_status:
            one_hot_labels = torch.cat([convert_one_hot(digits[:, 0]), convert_one_hot(digits[:, 1])], dim=1)
            loss = criterion(output, one_hot_labels)
            pred_0 = output[:, 0:10].detach().max(1)[1]
            pred_1 = output[:, 10:].detach().max(1)[1]
            pred = (pred_0 <= pred_1).long()
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
            one_hot_labels = torch.cat([convert_one_hot(digits[:, 0]), convert_one_hot(digits[:, 1])], dim=1)
            loss_1 = criterion(output, one_hot_labels)
            pred_0 = output[:, 0:10].detach().max(1)[1]
            pred_1 = output[:, 10:].detach().max(1)[1]
            pred = (pred_0 <= pred_1).long()
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
    if aux_labels_status:
        bits = 10
    else:
        bits = 2
    new_labels = torch.zeros((len(original_labels), bits))
    original_labels = original_labels.view(-1, 1)
    new_labels = new_labels.scatter(1, original_labels, 1)
    return new_labels


def start_learning(epoch = 25, w_sharing = 1, aux_labels = 1, print_train = 1, print_test = 1):
    global weight_sharing_status, aux_labels_status, print_train_data, print_test_data
    weight_sharing_status = w_sharing
    aux_labels_status = aux_labels
    print_train_data = print_train
    print_test_data = print_test
    print("=" * 100)
    print('Weight sharing:', bool(weight_sharing_status), '   Auxiliary losses:', bool(aux_labels_status))

    para_size = 0
    for i in net.parameters():
        para_size += torch.prod(torch.tensor(i.size()))
    print('Parameters quantity:', para_size.item() // 2, '\n')

    train_accuracy_list = []
    test_accuracy_list = []

    for i in range(epoch):
        train_loss,  train_accuracy = train(i + 1)
        train_accuracy_list.append(train_accuracy)
        test_loss, test_accuracy = test(i + 1)
        test_accuracy_list.append(test_accuracy)
    print("After %d epoch of training, the accuracy on test set is %f\n" % (epoch, max(test_accuracy_list)))


if __name__ == '__main__':
    start_learning(epoch = 25, w_sharing = 1, aux_labels = 1, print_train = 1, print_test = 1)
