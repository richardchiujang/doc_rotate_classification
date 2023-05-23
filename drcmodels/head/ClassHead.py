# -*- coding: utf-8 -*-

import torch.nn as nn

class ClassHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = in_channels // 4

        self.ClassNetwork = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.mid_channels * 56 * 56, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Linear(10, self.out_channels),
        )

    def forward(self, x):
        x = self.ClassNetwork(x)
        return x

