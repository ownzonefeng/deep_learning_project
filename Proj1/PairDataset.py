from torch.utils.data import Dataset


class PairDataset(Dataset):

    def __init__(self, data, train=True, aux_labels=False):
        # create the data set
        if train:
            # if this is the train data set, store data from 0
            idx = 0
        else:
            # if this is the test data set, store data from 3
            idx = 3
        self.features = data[idx]
        self.bool_labels = data[idx + 1]
        if aux_labels:
            self.digit_labels = data[idx + 2]
        self.aux_labels = aux_labels

    def __len__(self):
        # override the class method. return the length of data
        return len(self.bool_labels)

    def __getitem__(self, idx):
        # override the class method. return the item at the index(idx)
        if self.aux_labels:
            return self.features[idx], self.bool_labels[idx], self.digit_labels[idx]
        else:
            return self.features[idx], self.bool_labels[idx], self.bool_labels[idx]
