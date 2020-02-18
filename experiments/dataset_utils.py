#!/usr/bin/env python3

import os.path as osp
from torchvision import datasets
from torchvision import transforms
from torch.utils import data

dataset_choices = ['MNIST', 'FashionMNIST', 'SVHN', 'STL10', 'CIFAR10', 'CIFAR100', 'TinyImageNet', 'CelebA', 'LSUN']

def get_dataset(dataset_name, dataset_root, train_transform, test_transform):

    if dataset_name == 'MNIST':
        train_data = datasets.MNIST(osp.join(dataset_root, 'MNIST'), train=True, transform=train_transform, download=False)
        test_data = datasets.MNIST(osp.join(dataset_root, 'MNIST'), train=False, transform=test_transform, download=False)
    elif dataset_name == 'FashionMNIST':
        train_data = datasets.FashionMNIST(osp.join(dataset_root, 'FashionMNIST'), train=True, transform=train_transform, download=False)
        test_data = datasets.FashionMNIST(osp.join(dataset_root, 'FashionMNIST'), train=False, transform=test_transform, download=False)
    elif dataset_name == 'CIFAR10':
        train_data = datasets.CIFAR10(osp.join(dataset_root, 'CIFAR10'), train=True, transform=train_transform, download=False)
        test_data = datasets.CIFAR10(osp.join(dataset_root, 'CIFAR10'), train=False, transform=test_transform, download=False)
    elif dataset_name == 'CIFAR100':
        train_data = datasets.CIFAR100(osp.join(dataset_root, 'CIFAR100'), train=True, transform=train_transform, download=False)
        test_data = datasets.CIFAR100(osp.join(dataset_root, 'CIFAR100'), train=False, transform=test_transform, download=False)
    elif dataset_name == 'STL10':
        train_data = datasets.STL10(osp.join(dataset_root, 'STL10'), split='train', transform=train_transform, download=False)
        test_data = datasets.STL10(osp.join(dataset_root, 'STL10'), split='test', transform=test_transform, download=False)
    elif dataset_name == 'SVHN':
        train_data = datasets.SVHN(osp.join(dataset_root, 'SVHN'), split='train', transform=train_transform, download=False)
        test_data = datasets.SVHN(osp.join(dataset_root, 'SVHN'), split='test', transform=test_transform, download=False)
    elif dataset_name == 'TinyImageNet':
        train_data = datasets.ImageFolder(osp.join(dataset_root, 'tiny-imagenet-200', 'torch_train'), transform=train_transform)
        test_data = datasets.ImageFolder(osp.join(dataset_root, 'tiny-imagenet-200', 'torch_val'), transform=test_transform)
    elif dataset_name == 'CelebA':
        train_data = datasets.ImageFolder(osp.join(dataset_root, 'CelebA'), transform=train_transform)
        test_data = None
    elif dataset_name == 'LSUN':
        train_data = datasets.LSUN(osp.join(dataset_root, 'LSUN'), classes=['bedroom_train'], transform=train_transform)
        test_data = None
    else:
        raise Exception('Unknown dataset: {}'.format(dataset_name))

    return train_data, test_data

def get_mean_std(dataset):
    if dataset == 'MNIST':
        mean = (0.1307,)
        std = (0.3081,)
    elif dataset == 'FashionMNIST':
        mean = (0.5,)
        std  = (0.5,)
    elif dataset == 'CIFAR10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'CIFAR100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif dataset == 'STL10':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif dataset == 'SVHN':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif dataset == 'TinyImageNet':
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2302, 0.2265, 0.2262]
    elif dataset == 'CelebA':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif dataset == 'LSUN':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        raise Exception('Unknown dataset: {}'.format(dataset))

    return mean, std

def get_dataset_transforms(mean, std, size, augment=False):

    test_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ColorJitter(),
            # transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = test_transform

    return train_transform, test_transform

def get_dataset_config(dataset_name):

    config = dict()
    if dataset_name in ['MNIST', 'FashionMNIST']:

        config['size']        = 28
        config['ch']          = 1
        config['num_classes'] = 10
        # config['train']       = 6e4
        # config['test']        = 1e4
        config['dataset_size']= {'train': 6e4, 'test': 1e4}

    elif dataset_name in ['CIFAR10', 'CIFAR100']:

        config['size']        = 32
        config['ch']          = 3
        config['num_classes'] = 10
        if dataset_name == 'CIFAR100':
            config['num_classes'] = 100
        # config['train']       = 5e4
        # config['test']        = 1e4
        config['dataset_size']= {'train': 5e4, 'test': 1e4}

    elif dataset_name in ['STL10']:

        config['size']        = 96 
        config['ch']          = 3
        config['num_classes'] = 10
        # config['train']       = 5e4
        # config['test']        = 1e4
        config['dataset_size']= {'train': 5e4, 'test': 8e4}

    elif dataset_name in ['SVHN']:

        config['size']        = 32
        config['ch']          = 3
        config['num_classes'] = 11
        # config['train']       = 73257
        # config['test']        = 26032
        config['dataset_size']= {'train': 73257, 'test': 26032}

    elif dataset_name in ['TinyImageNet']:
        config['size']        = 64
        config['ch']          = 3
        config['num_classes'] = 200
        # config['train']       = 1e5
        # config['test']        = 1e4
        config['dataset_size']= {'train': 1e5, 'test': 1e4}

    elif dataset_name in ['CelebA']:
        config['size']        = 64
        config['ch']          = 3
        config['num_classes'] = None

        config['dataset_size']= {'train': 202599}

    elif dataset_name in ['LSUN']:
        config['size'] = 64
        config['ch']   = 3
        config['num_classes'] = None
        
        config['dataset_size'] = {'train': 168103}

    else:
        raise Exception('unknown dataset: {}'.format(dataset_name))

    return config