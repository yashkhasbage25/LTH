#!/bin/sh

python reruns_optimizer_lottery.py -lr 1e-1 -l2 1e-4 -n 100 -d FashionMNIST -b 128 -m LeNet -o SGD -reruns 5 -r "model=LeNet,dataset=FashionMNIST,opt,run1" -cuda 3 --augment 
python reruns_optimizer_lottery.py -lr 1e-1 -l2 1e-4 -n 100 -d FashionMNIST -b 128 -m LeNet -o Adam -reruns 5 -r "model=LeNet,dataset=FashionMNIST,opt,run2" -cuda 3 --augment 
python reruns_optimizer_lottery.py -lr 1e-1 -l2 1e-4 -n 100 -d FashionMNIST -b 128 -m LeNet -o AdamW -reruns 5 -r "model=LeNet,dataset=FashionMNIST,opt,run3" -cuda 3 --augment 
python reruns_optimizer_lottery.py -lr 1e-1 -l2 1e-4 -n 100 -d FashionMNIST -b 128 -m LeNet -o RMSprop -reruns 5 -r "model=LeNet,dataset=FashionMNIST,opt,run4" -cuda 3 --augment 

python reruns_optimizer_lottery.py -lr 1e-1 -l2 1e-4 -n 100 -d SVHN -b 128 -m LeNet -o SGD -reruns 5 -r "model=LeNet,dataset=SVHN,opt,run1" -cuda 3 --augment 
python reruns_optimizer_lottery.py -lr 1e-1 -l2 1e-4 -n 100 -d SVHN -b 128 -m LeNet -o Adam -reruns 5 -r "model=LeNet,dataset=SVHN,opt,run2" -cuda 3 --augment 
python reruns_optimizer_lottery.py -lr 1e-1 -l2 1e-4 -n 100 -d SVHN -b 128 -m LeNet -o AdamW -reruns 5 -r "model=LeNet,dataset=SVHN,opt,run3" -cuda 3 --augment 
python reruns_optimizer_lottery.py -lr 1e-1 -l2 1e-4 -n 100 -d SVHN -b 128 -m LeNet -o RMSprop -reruns 5 -r "model=LeNet,dataset=SVHN,opt,run4" -cuda 3 --augment 
