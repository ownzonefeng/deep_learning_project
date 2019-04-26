from SiameseNetworks_train import start_learning
import torch

all_round = 2
epoch = 2

train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []
train_time = []

log_file = open('train_records.txt', 'w')

for i in [0, 1]:
    for j in [0, 1]:
        print('\nWeight sharing:', bool(j), '   Auxiliary losses:', bool(i))
        print("=" * 100)

        log_file.write('\nWeight sharing:' + str(bool(j)) + '   Auxiliary losses:' + str(bool(i)))
        log_file.write("\n" + "=" * 100 + "\n")

        train_loss_temp = torch.zeros((all_round, epoch))
        train_accuracy_temp = torch.zeros((all_round, epoch))
        test_loss_temp = torch.zeros((all_round, epoch))
        test_accuracy_temp = torch.zeros((all_round, epoch))
        train_time_temp = torch.zeros((all_round, epoch))

        for r in range(all_round):
            train_loss_temp[r, :], train_accuracy_temp[r, :], test_loss_temp[r, :], test_accuracy_temp[r, :], \
                train_time_temp[r, :] = start_learning(epoch, j, i, 0, 0)
            train_accuracy_curr = torch.max(train_accuracy_temp[r, :]).item()
            test_accuracy_curr = torch.max(test_accuracy_temp[r, :]).item()
            train_time_curr = torch.sum(train_time_temp[r, :]).item()
            print('Round - {0:d}, train_accuracy {1:1.2%}, test accuracy {2:1.2%}, train time {3:2.2f}s'
                  .format(r + 1, train_accuracy_curr, test_accuracy_curr, train_time_curr))

            log_file.write('Round - {0:d}, train_accuracy {1:1.2%}, test accuracy {2:1.2%}, '
                           'train time {3:2.2f}s'.format(r + 1, train_accuracy_curr, test_accuracy_curr, train_time_curr))
            log_file.write('\n')

        train_loss.append(train_loss_temp)
        train_accuracy.append(train_accuracy_temp)
        test_loss.append(test_loss_temp)
        test_accuracy.append(test_accuracy_temp)
        train_time.append(train_time_temp)

        train_summary_mean = torch.mean(torch.max(train_accuracy_temp, dim=1)[0]).item()
        train_summary_std = torch.std(torch.max(train_accuracy_temp, dim=1)[0]).item()
        test_summary_mean = torch.mean(torch.max(test_accuracy_temp, dim=1)[0]).item()
        test_summary_std = torch.std(torch.max(test_accuracy_temp, dim=1)[0]).item()
        time_summary_mean = torch.mean(torch.sum(train_time_temp, dim=1)).item()
        time_summary_std = torch.std(torch.sum(train_time_temp, dim=1)).item()
        print('\nSummary:')
        print('train accuracy {0:1.2%} +/- {1:1.2%}'.format(train_summary_mean, train_summary_std))
        print('test accuracy {0:1.2%} +/- {1:1.2%}'.format(test_summary_mean, test_summary_std))
        print('train time {0:1.2f} +/- {1:1.2f}s'.format(time_summary_mean, time_summary_std))
        print('=' * 100)

        log_file.write('\nSummary:\n')
        log_file.write('train accuracy {0:1.2%} +/- {1:1.2%}\n'.format(train_summary_mean, train_summary_std))
        log_file.write('test accuracy {0:1.2%} +/- {1:1.2%}\n'.format(test_summary_mean, test_summary_std))
        log_file.write('train time {0:1.2f} +/- {1:1.2f}s\n'.format(time_summary_mean, time_summary_std))
        log_file.write("=" * 100 + "\n")


torch.save(train_loss, 'train_loss.pt')
torch.save(train_accuracy, 'train_accuracy.pt')
torch.save(test_loss, 'test_loss.pt')
torch.save(test_accuracy, 'test_accuracy.pt')
torch.save(train_time, 'train_time.pt')

log_file.close()
