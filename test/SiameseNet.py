import torch
from torch import nn
import torch.nn.functional as F


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvNet1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=0, stride=1),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, stride=2, padding=0),
                                      nn.Conv2d(32, 64, 3, padding=0, stride=1),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, stride=2, padding=0))

        self.FC1 = nn.Sequential(nn.Linear(256, 128),
                                 nn.Sigmoid(),
                                 nn.Linear(128, 64),
                                 nn.Sigmoid(),
                                 nn.Linear(64, 10),
                                 nn.Softmax(-1))

        self.ConvNet2 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=0, stride=1),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, stride=2, padding=0),
                                      nn.Conv2d(32, 64, 3, padding=0, stride=1),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, stride=2, padding=0))

        self.FC2 = nn.Sequential(nn.Linear(256, 128),
                                 nn.Sigmoid(),
                                 nn.Linear(128, 64),
                                 nn.Sigmoid(),
                                 nn.Linear(64, 10),
                                 nn.Softmax(-1))

    def forward(self, img_pair, w_share=False, aux_loss=False):
        conv_net_list = [self.ConvNet1, self.ConvNet1]
        fc_list = [self.FC1, self.FC1]
        output_list = []
        for i in range(2):
            img = torch.unsqueeze(img_pair[:, i], dim=1)
            output_i = conv_net_list[i](img)
            output_i = output_i.view(output_i.shape[0], -1)
            output_i = fc_list[i](output_i)
            output_list.append(output_i)
        output = F.pairwise_distance(output_list[0], output_list[1])
        return output
