#!/usr/bin/env python3

from . import mnist
from . import cifar
from . import svhn
from . import celeba

def get_model(model, dataset, utility='classifier'):

    if utility == 'classifier':
        if dataset in ['MNIST', 'FashionMNIST']:
            return getattr(mnist, model)()
        elif dataset in ['CIFAR10', 'CIFAR100']:
            return getattr(cifar, model)()
        elif dataset in ['SVHN']:
            return getattr(svhn, model)()
        else:
            raise Exception('unknown dataset: {}'.format(dataset))

    elif utility == 'gan':
        if dataset in ['MNIST', 'FashionMNIST']:
            disc = getattr(mnist, model).Discriminator()
            gene = getattr(mnist, model).Generator()
        elif dataset in ['CIFAR10', 'CIFAR100', 'SVHN']:
            disc = getattr(cifar, model).Discriminator()
            gene = getattr(cifar, model).Generator()     
        elif dataset in ['CelebA']:
            disc = getattr(celeba, model).Discriminator()
            gene = getattr(celeba, model).Generator()  
        else:
            raise Exception('unknown dataset: {}'.format(dataset)) 
        return disc, gene    
        
    else:
        raise Exception('unknown untility type: {}'.format(utility))