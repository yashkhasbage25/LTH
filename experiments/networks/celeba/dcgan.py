#!/usr/bin/env python3

import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, d=128):
        super(Generator, self).__init__()

        layers = [
            nn.ConvTranspose2d(100, d*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(d*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d*8, d*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d*4, d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d*2, d, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        ]

        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):

        return self.net(x)


class Discriminator(nn.Module):

    def __init__(self, d=128):
        super(Discriminator, self).__init__()

        layers = [
            nn.Conv2d(3, d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d, d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d*2, d*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d*4, d*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        ]

        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):

        return self.net(x)