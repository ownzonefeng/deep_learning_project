from SiameseNetworks_train import start_learning
import torch

all_round = 20  # the number of rounds for training
epoch = 25  # the number of epoch in each training

# initialize some lists to store data during the training
train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []
train_time = []

# the log file which records the process of training
log_file = open('train_records.txt', 'w')

for i in [0, 1]:
    for j in [0, 1]:
        # print current settings
        print('\nWeight sharing:', bool(i), '   Auxiliary losses:', bool(j))
        print("=" * 100)

        # record current setting
        log_file.write('\nWeight sharing:' + str(bool(i)) + '   Auxiliary losses:' + str(bool(j)))
        log_file.write("\n" + "=" * 100 + "\n")

        # initialize the tensors to store data of 20 rounds training in current setting
        train_loss_temp = torch.zeros((all_round, epoch))
        train_accuracy_temp = torch.zeros((all_round, epoch))
        test_loss_temp = torch.zeros((all_round, epoch))
        test_accuracy_temp = torch.zeros((all_round, epoch))
        train_time_temp = torch.zeros((all_round, epoch))

        for r in range(all_round):
            # start training. i controls weight sharing and j controls auxiliary loss
            # the training data in each epoch are saved in the tensors
            train_loss_temp[r, :], train_accuracy_temp[r, :], test_loss_temp[r, :], test_accuracy_temp[r, :], \
                train_time_temp[r, :] = start_learning(epoch, i, j, 0, 0)

            # the maximum accuracy in this round is the final accuracy
            train_accuracy_curr = torch.max(train_accuracy_temp[r, :]).item()
            test_accuracy_curr = torch.max(test_accuracy_temp[r, :]).item()

            # record the training time in this round
            train_time_curr = torch.sum(train_time_temp[r, :]).item()

            # print the performance of this round
            print('Round - {0:d}, train_accuracy {1:1.2%}, test accuracy {2:1.2%}, train time {3:2.2f}s'
                  .format(r + 1, train_accuracy_curr, test_accuracy_curr, train_time_curr))

            # record the performance of this round
            log_file.write('Round - {0:d}, train_accuracy {1:1.2%}, test accuracy {2:1.2%}, '
                           'train time {3:2.2f}s'.format(r + 1, train_accuracy_curr, test_accuracy_curr, train_time_curr))
            log_file.write('\n')

        # append the training data of current setting in the global list variables
        train_loss.append(train_loss_temp)
        train_accuracy.append(train_accuracy_temp)
        test_loss.append(test_loss_temp)
        test_accuracy.append(test_accuracy_temp)
        train_time.append(train_time_temp)

        # calculate some statistics in 20 rounds
        train_summary_mean = torch.mean(torch.max(train_accuracy_temp, dim=1)[0]).item()
        train_summary_std = torch.std(torch.max(train_accuracy_temp, dim=1)[0]).item()
        test_summary_mean = torch.mean(torch.max(test_accuracy_temp, dim=1)[0]).item()
        test_summary_std = torch.std(torch.max(test_accuracy_temp, dim=1)[0]).item()
        time_summary_mean = torch.mean(torch.sum(train_time_temp, dim=1)).item()
        time_summary_std = torch.std(torch.sum(train_time_temp, dim=1)).item()

        # print the summary in 20 rounds training
        print('\nSummary:')
        print('train accuracy {0:1.2%} +/- {1:1.2%}'.format(train_summary_mean, train_summary_std))
        print('test accuracy {0:1.2%} +/- {1:1.2%}'.format(test_summary_mean, test_summary_std))
        print('train time {0:1.2f}s +/- {1:1.2f}s'.format(time_summary_mean, time_summary_std))
        print('=' * 100)

        # log the summary in 20 rounds training
        log_file.write('\nSummary:\n')
        log_file.write('train accuracy {0:1.2%} +/- {1:1.2%}\n'.format(train_summary_mean, train_summary_std))
        log_file.write('test accuracy {0:1.2%} +/- {1:1.2%}\n'.format(test_summary_mean, test_summary_std))
        log_file.write('train time {0:1.2f}s +/- {1:1.2f}s\n'.format(time_summary_mean, time_summary_std))
        log_file.write("=" * 100 + "\n")

# write the training data onto disk
torch.save(train_loss, './Train data records/train_loss.pt')
torch.save(train_accuracy, './Train data records/train_accuracy.pt')
torch.save(test_loss, './Train data records/test_loss.pt')
torch.save(test_accuracy, './Train data records/test_accuracy.pt')
torch.save(train_time, './Train data records/train_time.pt')

# close log file
log_file.close()
