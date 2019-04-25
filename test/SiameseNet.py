import torch
from torch import nn
import torch.nn.functional as F


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvNet1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=0, stride=1),
                                      nn.Tanh(),
                                      nn.AvgPool2d(2, stride=2, padding=0),
                                      nn.Dropout(0.3),
                                      nn.Conv2d(64, 70, 3, padding=0, stride=1),
                                      nn.Tanh(),
                                      nn.AvgPool2d(2, stride=2, padding=0),
                                      nn.Dropout(0.3)
                                      )

        self.FC1 = nn.Sequential(nn.Linear(280, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 10),
                                 nn.Tanh())

        self.ConvNet2 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=0, stride=1),
                                      nn.Tanh(),
                                      nn.AvgPool2d(2, stride=2, padding=0),
                                      nn.Dropout(0.3),
                                      nn.Conv2d(64, 70, 3, padding=0, stride=1),
                                      nn.Tanh(),
                                      nn.AvgPool2d(2, stride=2, padding=0),
                                      nn.Dropout(0.3)
                                      )

        self.FC2 = nn.Sequential(nn.Linear(280, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 10),
                                 nn.Tanh())

        self.Combine = nn.Sequential(nn.Linear(20, 2), nn.Sigmoid())

    def forward(self, img_pair, w_share=False, aux_loss=False):
        if w_share:
            conv_net_list = [self.ConvNet1, self.ConvNet1]
            fc_list = [self.FC1, self.FC1]
        else:
            conv_net_list = [self.ConvNet1, self.ConvNet2]
            fc_list = [self.FC1, self.FC2]

        output_list = []
        for i in range(2):
            img = torch.unsqueeze(img_pair[:, i], dim=1)
            output_i = conv_net_list[i](img)
            output_i = output_i.view(output_i.shape[0], -1)
            output_i = fc_list[i](output_i)
            output_list.append(output_i)
        output = torch.cat(output_list, dim=1)
        output = self.Combine(output)
        return output
