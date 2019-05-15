import dlc_practical_prologue as prologue
from SiameseNet import SiameseNet
from PairDataset import PairDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import time

# define some variables needed in the training

# data set object and data loader object
data_train, data_test, data_train_loader, data_test_loader = [None] * 4

# variables control weight sharing, auxiliary loss, print train information, print test information
weight_sharing_status, aux_labels_status, print_train_data, print_test_data = [None] * 4

# network, criterion, optimizer variable
net, criterion, optimizer = [None] * 3


def train(epoch):
    net.train()  # change the net to train status
    total_loss = 0.0  # record cumulative loss in this epoch
    total_correct = 0.0  # record cumulative accuracy in this epoch

    for i, (images, labels, digits) in enumerate(data_train_loader):
        # set the gradient to zero
        optimizer.zero_grad()
        # train the network in this batch
        output = net(images, w_share=weight_sharing_status, aux_loss=aux_labels_status)

        if aux_labels_status:  # when auxiliary loss is enabled
            # convert boolean labels and digit labels into one hot key.
            one_hot_labels_digits_0 = convert_one_hot(digits[:, 0])
            one_hot_labels_digits_1 = convert_one_hot(digits[:, 1])
            one_hot_labels_bool = convert_one_hot(labels)

            # evaluate the loss function
            loss = criterion(output[0][:, 0:10], one_hot_labels_digits_0)
            loss += criterion(output[0][:, 10:], one_hot_labels_digits_1)
            loss += criterion(output[1], one_hot_labels_bool)

            # get the prediction
            pred = output[1].detach().max(1)[1]
        else:
            # convert boolean labels and into one hot key.
            one_hot_labels = convert_one_hot(labels)

            # evaluate the loss function
            loss = criterion(output, one_hot_labels)

            # get the prediction
            pred = output.detach().max(1)[1]

        total_loss += loss.detach().item()  # add the loss in current batch
        total_correct += pred.eq(labels.view_as(pred)).sum()  # add the accuracy in current batch

        loss.backward()  # back propagation
        optimizer.step()  # take a step
    avg_loss = total_loss / len(data_train)  # the average loss in this epoch
    accuracy = float(total_correct) / len(data_train)  # the average accuracy in this epoch

    # print_train_data decides whether to print information
    if print_train_data:
        print('Epoch: %d, Train Avg. Loss: %f, Accuracy: %f' % (epoch, avg_loss, accuracy))
    return avg_loss, accuracy


def test(epoch):
    net.eval()  # change the net to evaluation status
    total_loss = 0.0  # record total loss in this epoch
    total_correct = 0.0  # record total accuracy in this epoch
    for i, (images, labels, digits) in enumerate(data_test_loader):
        # get the output of test batch
        output = net(images, w_share=weight_sharing_status, aux_loss=aux_labels_status)

        if aux_labels_status:  # when auxiliary loss is enabled
            # convert boolean labels and digit labels into one hot key.
            one_hot_labels_digits_0 = convert_one_hot(digits[:, 0])
            one_hot_labels_digits_1 = convert_one_hot(digits[:, 1])
            one_hot_labels_bool = convert_one_hot(labels)

            # evaluate the loss function
            loss_1 = criterion(output[0][:, 0:10], one_hot_labels_digits_0)
            loss_1 += criterion(output[0][:, 10:], one_hot_labels_digits_1)
            loss_1 += criterion(output[1], one_hot_labels_bool)

            # get the prediction
            pred = output[1].detach().max(1)[1]
        else:
            # convert boolean labels and into one hot key.
            one_hot_labels = convert_one_hot(labels)

            # evaluate the loss function
            loss_1 = criterion(output, one_hot_labels)

            # get the prediction
            pred = output.detach().max(1)[1]

        total_loss += loss_1  # add the loss in current batch
        total_correct += pred.eq(labels.view_as(pred)).sum()  # add the accuracy in current batch

    avg_loss = float(total_loss) / len(data_test)  # the average loss in this epoch
    accuracy = float(total_correct) / len(data_test)  # the average accuracy in this epoch

    # print_test_data decides whether to print information
    if print_test_data:
        print('Epoch: %d, Test  Avg. Loss: %f, Accuracy: %f' % (epoch, avg_loss, accuracy))
    if print_train_data and print_test_data:
        print('\n')
    return avg_loss, accuracy


def convert_one_hot(original_labels):
    bits = len(torch.unique(original_labels))  # get the number of unique keys

    # if the number of unique keys is greater than 2, the bits is set to 10.
    # This is to handle some accidents, such as only 9 number in a batch.
    # It is not likely that there are only two number in a batch, so we ignore this extreme condition
    if bits > 2:
        bits = 10

    # convert the label into one hot key
    new_labels = torch.zeros((len(original_labels), bits))
    original_labels = original_labels.view(-1, 1)
    new_labels = new_labels.scatter(1, original_labels, 1)
    return new_labels


def weight_init(module):
    # this function is to initialize weight when needed
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        module.reset_parameters()


def data_generation():
    # this function is to generate train and test data
    global data_train, data_test, data_train_loader, data_test_loader
    # using helper to generate 1000 pairs of train data and test data
    data = prologue.generate_pair_sets(1000)

    # create torch style data set
    data_train = PairDataset(data, train=True, aux_labels=True)
    data_test = PairDataset(data, train=False, aux_labels=True)

    # create data loader
    data_train_loader = DataLoader(data_train, batch_size=100, shuffle=True, num_workers=12)
    data_test_loader = DataLoader(data_test, batch_size=100, num_workers=12)


def start_learning(epoch=25, w_sharing=1, aux_labels=1, print_train=1, print_test=1):
    # get all parameters in this training
    global weight_sharing_status, aux_labels_status, print_train_data, print_test_data
    weight_sharing_status = w_sharing
    aux_labels_status = aux_labels
    print_train_data = print_train
    print_test_data = print_test

    # uncomment following two lines to enable print parameters information
    # print("=" * 100)
    # print('Weight sharing:', bool(weight_sharing_status), '   Auxiliary losses:', bool(aux_labels_status))

    # initialize network, criterion, optimizer
    global net, criterion, optimizer
    net = SiameseNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-2)

    # generate needed data and initialize all weights
    data_generation()
    net.apply(weight_init)

    # this block is to compute the number of parameters of network
    para_size = 0
    for i in net.parameters():
        para_size += torch.prod(torch.tensor(i.size()))
    # uncomment the following line to print the number of parameters of network
    # print('Parameters quantity:', para_size.item() // 2, '\n')

    # initialize tensors to record data of each epoch during the training
    train_accuracy_list = torch.empty((1, epoch))
    train_loss_list = torch.empty((1, epoch))
    test_accuracy_list = torch.empty((1, epoch))
    test_loss_list = torch.empty((1, epoch))
    train_time_list = torch.empty((1, epoch))

    # train the network in give epochs
    for i in range(epoch):
        s = time.perf_counter()  # start time counter
        train_loss, train_accuracy = train(i + 1)  # get the train loss and train accuracy
        elapse = time.perf_counter() - s  # count the training time
        test_loss, test_accuracy = test(i + 1)  # get the test loss and test accuracy

        # record data in this epoch
        train_loss_list[0, i] = train_loss
        train_accuracy_list[0, i] = train_accuracy
        test_loss_list[0, i] = test_loss
        test_accuracy_list[0, i] = test_accuracy
        train_time_list[0, i] = elapse

    # reset network, criterion, optimizer after finishing the training
    net, criterion, optimizer = [None] * 3

    # reset data set, data loader after finishing the training
    global data_train, data_test, data_train_loader, data_test_loader
    data_train, data_test, data_train_loader, data_test_loader = [None] * 4

    # return the data in this training
    return train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list, train_time_list


if __name__ == '__main__':
    _ = start_learning(epoch=25, w_sharing=0, aux_labels=0, print_train=1, print_test=1)
