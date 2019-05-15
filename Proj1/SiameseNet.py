import torch
from torch import nn


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()

        # sub-network 1: the structure of convolution net
        self.ConvNet1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=0, stride=1),
                                      nn.Tanh(),
                                      nn.AvgPool2d(2, stride=2, padding=0),
                                      nn.Dropout(0.3),
                                      nn.Conv2d(32, 64, 3, padding=0, stride=1),
                                      nn.Tanh(),
                                      nn.AvgPool2d(2, stride=2, padding=0),
                                      nn.Dropout(0.3)
                                      )
        # sub-network 1: the structure of fully connected net
        self.FC1 = nn.Sequential(nn.Linear(256, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 10),
                                 nn.Softmax(-1)
                                 )

        # sub-network 2: the structure of convolution net
        self.ConvNet2 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=0, stride=1),
                                      nn.Tanh(),
                                      nn.AvgPool2d(2, stride=2, padding=0),
                                      nn.Dropout(0.3),
                                      nn.Conv2d(32, 64, 3, padding=0, stride=1),
                                      nn.Tanh(),
                                      nn.AvgPool2d(2, stride=2, padding=0),
                                      nn.Dropout(0.3)
                                      )
        # sub-network 2: the structure of fully connected net
        self.FC2 = nn.Sequential(nn.Linear(256, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 10),
                                 nn.Softmax(-1)
                                 )

        # combination layers
        self.Combine = nn.Sequential(nn.Linear(20, 2), nn.Softmax(-1))

    def forward(self, img_pair, w_share=True, aux_loss=True):
        if w_share:
            # if we use weight sharing, two channels use the parameters of sub-network 1
            conv_net_list = [self.ConvNet1, self.ConvNet1]
            fc_list = [self.FC1, self.FC1]
        else:
            # if we do not use weight sharing, two channels use different sub-network
            conv_net_list = [self.ConvNet1, self.ConvNet2]
            fc_list = [self.FC1, self.FC2]

        output_list = []
        # training with two channel images
        for i in range(2):
            # each channel passed through given sub-networks
            img = torch.unsqueeze(img_pair[:, i], dim=1)
            output_i = conv_net_list[i](img)
            output_i = output_i.view(output_i.shape[0], -1)
            output_i = fc_list[i](output_i)
            output_list.append(output_i)

        if aux_loss:
            # if we use auxiliary loss, the outputs of sub-networks are also returned
            aux_output = torch.cat(output_list, dim=1)
            bool_output = self.Combine(aux_output)
            return aux_output, bool_output
        else:
            # if we do not use auxiliary loss, only the final layer can yield output
            output = torch.cat(output_list, dim=1)
            output = self.Combine(output)
            return output
