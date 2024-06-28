import torch
import torch.nn as nn



class PointDecoder(nn.Module):
    def __init__(self, channels, residual=True, use_bn=True):
        super().__init__()

        self.residual = residual
        self.mlps = nn.ModuleList()

        for i in range(len(channels) - 1):
            modules = []
            if i == 0 or not self.residual:
                modules.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=1))
            else:
                modules.append(nn.Conv1d(channels[i] + channels[0], channels[i + 1], kernel_size=1))

            if i != len(channels) - 1:
                if use_bn:
                    modules.append(nn.BatchNorm1d(channels[i + 1]))
                modules.append(nn.LeakyReLU(inplace=True))

            self.mlps.append(nn.Sequential(*modules))

    def forward(self, x):
        x_ = x
        for i, m in enumerate(self.mlps):
            if i != 0 and self.residual:
                x_ = torch.cat([x_, x], dim=1)
            x_ = m(x_)
        return x_
