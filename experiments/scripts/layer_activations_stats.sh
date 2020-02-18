#!/bin/bash

python layer_activations_stats.py -m ResNet18 -d CIFAR10 -b 6000 -r "model=ResNet18,dataset=CIFAR10,rew,run2" -cuda 0 -relu_only
python layer_activations_stats.py -m ResNet18 -d CIFAR10 -b 6000 -r "model=ResNet18,dataset=CIFAR10,rew,run2" -cuda 0 

python layer_activations_stats.py -m VGG16_bn -d SVHN -b 6000 -r "model=VGG16_bn,dataset=SVHN,run1" -cuda 0 -relu_only
python layer_activations_stats.py -m VGG16_bn -d SVHN -b 6000 -r "model=VGG16_bn,dataset=SVHN,run1" -cuda 0 

