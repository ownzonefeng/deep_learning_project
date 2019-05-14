import torch
from torch import nn


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvNet1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=0, stride=1),
                                      nn.Tanh(),
                                      nn.AvgPool2d(2, stride=2, padding=0),
                                      nn.Dropout(0.3),
                                      nn.Conv2d(32, 64, 3, padding=0, stride=1),
                                      nn.Tanh(),
                                      nn.AvgPool2d(2, stride=2, padding=0),
                                      nn.Dropout(0.3)
                                      )

        self.FC1 = nn.Sequential(nn.Linear(256, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 10),
                                 nn.Softmax(-1)
                                 )

        self.ConvNet2 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=0, stride=1),
                                      nn.Tanh(),
                                      nn.AvgPool2d(2, stride=2, padding=0),
                                      nn.Dropout(0.3),
                                      nn.Conv2d(32, 64, 3, padding=0, stride=1),
                                      nn.Tanh(),
                                      nn.AvgPool2d(2, stride=2, padding=0),
                                      nn.Dropout(0.3)
                                      )

        self.FC2 = nn.Sequential(nn.Linear(256, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 10),
                                 nn.Softmax(-1)
                                 )

        # self.ConvNet1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=0, stride=1),
        #                              nn.Tanh(),
        #                              nn.AvgPool2d(2, stride=2, padding=0),
        #                              nn.Conv2d(32, 64, 3, padding=0, stride=1),
        #                               nn.Tanh(),
        #                              nn.AvgPool2d(2, stride=2, padding=0))
        #
        # self.FC1 = nn.Sequential(nn.Linear(256, 128),
        #                          nn.Tanh(),
        #                         nn.Linear(128, 10),
        #                         nn.Softmax(-1))
        #
        # self.ConvNet2 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=0, stride=1),
        #                               nn.Tanh(),
        #                              nn.AvgPool2d(2, stride=2, padding=0),
        #                              nn.Conv2d(32, 64, 3, padding=0, stride=1),
        #                               nn.Tanh(),
        #                              nn.AvgPool2d(2, stride=2, padding=0))
        #
        # self.FC2 = nn.Sequential(nn.Linear(256, 128),
        #                          nn.Tanh(),
        #                         nn.Linear(128, 10),
        #                         nn.Softmax(-1))

        self.Combine = nn.Sequential(nn.Linear(20, 2), nn.Softmax(-1))

    def forward(self, img_pair, w_share=True, aux_loss=True):
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

        if aux_loss:
            aux_output = torch.cat(output_list, dim=1)
            bool_output = self.Combine(aux_output)
            return aux_output, bool_output
        else:
            output = torch.cat(output_list, dim=1)
            output = self.Combine(output)
            return output
