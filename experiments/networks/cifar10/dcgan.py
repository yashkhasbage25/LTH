#!/usr/bin/env python3

import torch
import torch.nn as nn

class Generator(nn.Module):
    
    def __init__(self, d=64, z_len=100):

        super(Generator, self).__init__()

        layers = [
            nn.ConvTranspose2d(z_len, d * 8, 4, 1, 0), # (8d, 4, 4)
            nn.BatchNorm2d(d * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1), # (4d, 8, 8)
            nn.BatchNorm2d(d * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1), # (2d, 16, 16)
            nn.BatchNorm2d(d * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(d * 2, d, 4, 2, 1), # (d, 32, 32)
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            nn.ConvTranspose2d(d, 3, 3, 1, 1), # (3, 32, 32)
            nn.Tanh()
        ]

        self.net = nn.Sequential(*layers)

        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.init.normal_(m.weight.data, 1.0, 0.02)
                m.init.constant_(m.bias.data, 0)

    def forward(self, x):
        
        return self.net(x)


class Discriminator(nn.Module):

    def __init__(self, d=64):

        super(Discriminator, self).__init__()

        layers = [
            nn.Conv2d(3, d, 4, 2, 1), # (d, 16, 16)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d, d * 2, 4, 2, 1), # (2d, 8, 8)
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d * 2, d * 4, 4, 2, 1), # (4d, 4, 4)
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d * 4, d * 8, 4, 2, 1), # (8d, 2, 2)
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d * 8, 1, 4, 2, 1), # (1, 1, 1)
            nn.Sigmoid()
        ]

        self.net = nn.Sequential(*layers)

        for m in self._modules:
            if isinstance(m, nn.Conv2d):
                m.init.normal_(0.0, 0.02)
            
    def forward(self, x):

        return self.net(x)