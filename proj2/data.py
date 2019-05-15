from torch import empty
import math

class dataloader():

    def __init__(self, batchsize):
        self.x = empty((1000, 2)).uniform_()
        d = (self.x[:, 0] - 0.5) ** 2 + (self.x[:, 1] - 0.5) ** 2
        self.label_1d = (d < (1 / (2 * math.pi))).float().reshape((1000, 1))  # 0 and 1 labels
        self.label_2d = (self.label_1d - 0.5) * 2
        self.label_2d = self.label_2d.expand(-1, 2)  # (-1, -1) is label 0. (1, 1) is label 1

        self.batchsize = batchsize
        self.n = 0

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        start = self.n
        end = self.n + self.batchsize
        if start==1000:
            raise StopIteration
        else:
            self.n += self.batchsize
            return self.x[start:end], self.label_1d[start:end], self.label_2d[start:end]

    def next(self):
        start = self.n
        end = self.n + self.batchsize
        if start==1000:
            raise StopIteration
        else:
            self.n += self.batchsize
            return self.x[start:end], self.label_1d[start:end], self.label_2d[start:end]

    def __len__(self):
        return 1000/self.batchsize


def label_2dto1d(label_2d):
    '''

    :param label_2d: shape(size, 2)
    :return: label_1d: shape(size, 1)
    '''
    label_1d = label_2d[:,0] + label_2d[:,1]
    label_1d = label_1d.reshape((-1, 1))
    label_1d = (label_1d>0).float()
    return label_1d
