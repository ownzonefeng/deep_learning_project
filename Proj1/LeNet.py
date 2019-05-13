from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.ConvNet = nn.Sequential(nn.Conv2d(1, 32, 3, padding=0, stride=1),
                                     nn.ReLU(),
                                     nn.AvgPool2d(2, stride=2, padding=0),
                                     nn.Conv2d(32, 64, 3, padding=0, stride=1),
                                     nn.ReLU(),
                                     nn.AvgPool2d(2, stride=2, padding=0))

        self.FC = nn.Sequential(nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, 10),
                                nn.Softmax(-1))

        self.ConvNet2 = nn.Sequential(nn.Conv2d(1, 6, 3, padding=0, stride=1),
                                     nn.Tanh(),
                                     nn.AvgPool2d(2, stride=2, padding=0),
                                     nn.Conv2d(6, 16, 3, padding=0, stride=1),
                                     nn.Tanh(),
                                     nn.AvgPool2d(2, stride=2, padding=0))

        self.FC2 = nn.Sequential(nn.Linear(64, 32),
                                nn.Tanh(),
                                nn.Linear(32, 10),
                                nn.Softmax(-1))

    def forward(self, img):
        output = self.ConvNet(img)
        output = output.view(output.shape[0], -1)
        output = self.FC(output)
        return output