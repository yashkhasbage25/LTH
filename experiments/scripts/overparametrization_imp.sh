#!/bin/sh

python overparametrization_imp.py -lr 1e-1 -l2 1e-4 -d FashionMNIST -b 128 -m FC -n_hidden 3 -hidden_dim 100 -act ReLU -r "model=FC,dataset=FashionMNIST,ovp,run1" -cuda 3 --augment --milestones 50 75 -f
python overparametrization_imp.py -lr 1e-1 -l2 1e-4 -d FashionMNIST -b 128 -m FC -n_hidden 6 -hidden_dim 100 -act ReLU -r "model=FC,dataset=FashionMNIST,ovp,run2" -cuda 3 --augment --milestones 50 75 -f
python overparametrization_imp.py -lr 1e-1 -l2 1e-4 -d FashionMNIST -b 128 -m FC -n_hidden 3 -hidden_dim 200 -act ReLU -r "model=FC,dataset=FashionMNIST,ovp,run3" -cuda 3 --augment --milestones 50 75 -f

python overparametrization_imp.py -lr 1e-1 -l2 1e-4 -d FashionMNIST -b 128 -m Conv -n_hidden 3 -width 5 -act ReLU -r "model=Conv,dataset=FashionMNIST,ovp,run1" -cuda 3 --augment --milestones 50 75 -f
python overparametrization_imp.py -lr 1e-1 -l2 1e-4 -d FashionMNIST -b 128 -m Conv -n_hidden 6 -width 5 -act ReLU -r "model=Conv,dataset=FashionMNIST,ovp,run2" -cuda 3 --augment --milestones 50 75 -f
python overparametrization_imp.py -lr 1e-1 -l2 1e-4 -d FashionMNIST -b 128 -m Conv -n_hidden 3 -width 10 -act ReLU -r "model=Conv,dataset=FashionMNIST,ovp,run3" -cuda 3 --augment --milestones 50 75 -f

python overparametrization_imp.py -lr 1e-1 -l2 1e-4 -d SVHN -b 128 -m Conv -n_hidden 3 -width 20 -act ReLU -r "model=Conv,dataset=SVHN,ovp,run1" -cuda 3 --augment --milestones 50 75 -f
python overparametrization_imp.py -lr 1e-1 -l2 1e-4 -d SVHN -b 128 -m Conv -n_hidden 5 -width 20 -act ReLU -r "model=Conv,dataset=SVHN,ovp,run2" -cuda 3 --augment --milestones 50 75 -f
python overparametrization_imp.py -lr 1e-1 -l2 1e-4 -d SVHN -b 128 -m Conv -n_hidden 3 -width 40 -act ReLU -r "model=Conv,dataset=SVHN,ovp,run3" -cuda 3 --augment --milestones 50 75 -f
