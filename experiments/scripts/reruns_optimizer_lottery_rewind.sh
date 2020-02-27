#!/bin/sh

python3 reruns_optimizer_lottery_rewind.py -lr 1e-2 -l2 1e-4 -n 200 -d CIFAR10 -b 128 -m VGG11_bn -o Adam -reruns 5 -r "model=VGG11_bn,dataset=CIFAR10,opt,run1" -rewind 15 -cuda 3 --augment --milestones 100 150 
python3 reruns_optimizer_lottery_rewind.py -lr 1e-2 -l2 1e-4 -n 200 -d CIFAR10 -b 128 -m VGG11_bn -o AdamW -reruns 5 -r "model=VGG11_bn,dataset=CIFAR10,opt,run2" -rewind 15 -cuda 3 --augment --milestones 100 150 
python3 reruns_optimizer_lottery_rewind.py -lr 1e-2 -l2 1e-4 -n 200 -d CIFAR10 -b 128 -m VGG11_bn -o SGD -reruns 5 -r "model=VGG11_bn,dataset=CIFAR10,opt,run3" -rewind 15 -cuda 3 --augment --milestones 100 150 
python3 reruns_optimizer_lottery_rewind.py -lr 1e-2 -l2 1e-4 -n 200 -d CIFAR10 -b 128 -m VGG11_bn -o RMSprop -reruns 5 -r "model=VGG11_bn,dataset=CIFAR10,opt,run4" -rewind 15 -cuda 3 --augment --milestones 100 150 


python3 reruns_optimizer_lottery_rewind.py -lr 1e-2 -l2 1e-4 -n 200 -d CIFAR10 -b 128 -m ResNet18 -reruns 5 -r "model=ResNet18,dataset=CIFAR10,opt,run1" -rewind 15 -cuda 3 --augment --milestones 100 150 
