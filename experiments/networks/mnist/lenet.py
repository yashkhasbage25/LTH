#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):

    def __init__(self, ch=1, size=28):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(ch, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1   = nn.Linear(16*5*5, 120)
        # self.fc2   = nn.Linear(120, 84)
        # self.fc3   = nn.Linear(84, 10)
        classfier_input_size = self.dummy_forward(torch.zeros(10, ch, size, size)).shape[-1]
        self.classifier = nn.Sequential(
            nn.Linear(classfier_input_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def dummy_forward(self, x):
        with torch.no_grad():
            out = F.relu(self.conv1(x))
            out = F.max_pool2d(out, 2)
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)     
        return out   

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)
        out = self.classifier(out)
        return out


class LeNetSigmoid(nn.Module):

    def __init__(self, ch=1, size=28):

        super(LeNetSigmoid, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(ch, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2)
        )

        classfier_input_size = self.dummy_forward(torch.zeros(10, ch, size, size)).shape[-1]
        
        self.classifier = nn.Sequential(
            nn.Linear(classfier_input_size, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def dummy_forward(self, x):
        with torch.no_grad():
            out = self.features(x)  
            out = out.view(out.size(0), -1)
        return out   

    def forward(self, x):

        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out


class LeNetTanh(nn.Module):
    
    def __init__(self, ch=1, size=28):

        super(LeNetTanh, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(ch, 6, 5),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),
            nn.MaxPool2d(2)
        )

        classfier_input_size = self.dummy_forward(torch.zeros(10, ch, size, size)).shape[-1]
        
        self.classifier = nn.Sequential(
            nn.Linear(classfier_input_size, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )

    def dummy_forward(self, x):
        with torch.no_grad():
            out = self.features(x)  
            out = out.view(out.size(0), -1)
        return out   

    def forward(self, x):

        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out


class LeNetELU(nn.Module):
    
    def __init__(self, ch=1, size=28):

        super(LeNetELU, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(ch, 6, 5),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ELU(),
            nn.MaxPool2d(2)
        )

        classfier_input_size = self.dummy_forward(torch.zeros(10, ch, size, size)).shape[-1]
        
        self.classifier = nn.Sequential(
            nn.Linear(classfier_input_size, 120),
            nn.ELU(),
            nn.Linear(120, 84),
            nn.ELU(),
            nn.Linear(84, 10)
        )

    def dummy_forward(self, x):
        with torch.no_grad():
            out = self.features(x)  
            out = out.view(out.size(0), -1)
        return out   

    def forward(self, x):

        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out