import torch
import torch.nn as nn
from torch.nn import functional as F
from .sMLP_block import *


def _SplitChannels(channels, num_groups):
    split_channels = [channels//num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels


class FastMixBlock(nn.Module):
    """
    modified from https://github.com/romulus0914/MixNet-PyTorch/blob/master/mixnet.py
    """

    def __init__(self, in_chan, out_chan):
        super(FastMixBlock, self).__init__()
        kernel_size = [1, 3, 5, 7]
        self.num_groups = len(kernel_size)
        self.split_in_channels = _SplitChannels(in_chan, self.num_groups)
        self.split_out_channels = _SplitChannels(out_chan, self.num_groups)

        self.grouped_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.grouped_conv.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.split_in_channels[i],
                        self.split_out_channels[i],
                        kernel_size[i],
                        stride=1,
                        padding=(kernel_size[i] - 1) // 2,
                        bias=True
                    ),
                    nn.BatchNorm2d(self.split_out_channels[i]),
                    nn.PReLU()
                )
            )

    def forward(self, x):
        if self.num_groups == 1:
            return self.grouped_conv[0](x)

        x_split = torch.split(x, self.split_in_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.grouped_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class sff(nn.Module):
    def __init__(self, in_chan, reduction_ratio=4):
        super(sff, self).__init__()
        self.mix1 = FastMixBlock(in_chan, in_chan)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()


        self.mlp1 = nn.Sequential(
            Flatten(),
            nn.Linear(in_chan, in_chan // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_chan // reduction_ratio, in_chan)
            )

        self.mlp2 = nn.Sequential(
            Flatten(),
            nn.Linear(in_chan, in_chan // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_chan // reduction_ratio, in_chan)
            )

        self.mlp3 = sMLPBlock2(W=256, H=256,channels=in_chan)


    def forward(self, x):
        shortcut = x
        mix1 = self.mix1(x)

        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_avg = self.mlp1(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x)

        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_max = self.mlp2(max_pool).unsqueeze(2).unsqueeze(3).expand_as(x)

        channel_att_ms = self.mlp3(mix1)

        avg_con = channel_att_ms * channel_att_avg
        max_con = channel_att_ms * channel_att_max

        out1 = avg_con + shortcut
        out2 = max_con + shortcut

        return self.relu1(out1),self.relu2(out2),mix1


