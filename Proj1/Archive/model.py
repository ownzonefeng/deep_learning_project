import torch
from torch import nn
class direct_model(nn.Module):
    def __init__(self):
        super(direct_model, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1)),
            nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1)) 
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.feature(x)
        y = y.view((x.shape[0], -1))
        y = self.classifier(y)
        return y