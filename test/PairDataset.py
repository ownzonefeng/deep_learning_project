from torch.utils.data import Dataset


class PairDataset(Dataset):

    def __init__(self, data, train=True, aux_labels=False):
        if train:
            idx = 0
        else:
            idx = 3
        self.features = data[idx]
        self.bool_labels = data[idx + 1]
        if aux_labels:
            self.digit_labels = data[idx + 2]
        self.aux_labels = aux_labels

    def __len__(self):
        return len(self.bool_labels)

    def __getitem__(self, idx):
        if self.aux_labels:
            return self.features[idx], self.digit_labels[idx]
        else:
            return self.features[idx], self.bool_labels[idx]
