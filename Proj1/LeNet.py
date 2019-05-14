from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.ConvNet = nn.Sequential(nn.Conv2d(1, 6, 5, padding=0, stride=1),
                                     nn.Tanh(),
                                     nn.AvgPool2d(2, stride=2, padding=0),
                                     nn.Conv2d(6, 16, 5, padding=0, stride=1),
                                     nn.Tanh())

        self.FC = nn.Sequential(nn.Linear(16, 14),
                                nn.Tanh(),
                                nn.Linear(14, 10),
                                nn.Softmax(-1))

    def forward(self, img):
        output = self.ConvNet(img)
        output = output.view(output.shape[0], -1)
        output = self.FC(output)
        return output
