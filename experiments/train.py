#!/usr/bin/env python3

import os
import sys
import tqdm
import torch
import shutil
import argparse
import numpy as np
import pickle as pkl
import seaborn as sns
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision import models
from torch.utils.tensorboard import SummaryWriter

import networks
from dataset_utils import *
from logging_utils import *

def parse_args():

    file_purpose = '''
    train a network
    '''

    parser = argparse.ArgumentParser(description=file_purpose,
        epilog=file_purpose, 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    default_lr = 1e-3
    default_l2 = 0.0
    default_num_epochs = 100
    default_batch_size = 64
    default_workers = 2
    default_dataset_root = osp.join(osp.dirname(os.getcwd()) ,'datasets')
    default_seed = 0
    default_momentum = 0.9


    parser.add_argument('-lr', type=float, default=default_lr, help='learning rate')
    parser.add_argument('-l2', type=float, default=default_l2, help='l2 penalty')
    parser.add_argument('-n', '--num_epochs', type=int, default=default_num_epochs, help='number of training epochs')
    parser.add_argument('-d', '--dataset', type=str, choices=dataset_choices, required=True, help='dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=default_batch_size, help='batch size for training')
    parser.add_argument('-j', '--workers', type=int, default=default_workers, help='number of wrokers for dataloader')
    parser.add_argument('-m', '--model', type=str, required=True, help='model')
    parser.add_argument('-f', action='store_true', help='force rewrite')
    parser.add_argument('-r', '--run', type=str, required=True, help='run directory prefix')
    parser.add_argument('-dp', action='store_true', help='data parallel model')
    parser.add_argument('-pre', action='store_true', help='pretrained imagenet weights')
    parser.add_argument('-mom', type=float, default=default_momentum, help='momentum for SGD')
    parser.add_argument('--cuda', type=int, help='use cuda, if use, then give gpu number')
    parser.add_argument('--seed', type=int, default=default_seed, help='seed for randomness')
    parser.add_argument('--augment', action='store_true', help='augment data with random-flip and random crop')
    parser.add_argument('--dataset_root', type=str, default=default_dataset_root, help='directory for dataset')
    parser.add_argument('-pdb', '--with_pdb', action='store_true', help='run with python debugger')

    return parser.parse_args()


def evaluate_model(model, 
                   criterion, 
                   dataloader, 
                   device, 
                   dataset_size
                   ):

    model.eval()
    running_loss = 0.0
    running_corrects = 0

    # iterate over data
    with torch.no_grad():
        for batch, truth in dataloader:
            batch = batch.to(device)
            truth = truth.to(device)

            output = model(batch)
            _, preds = torch.max(output, 1)
            running_corrects += torch.sum(preds == truth)
                
            loss = criterion(output, truth)

            # accummulate loss
            running_loss += loss.item() * batch.size(0)
        
    final_loss = running_loss / dataset_size
    final_acc = running_corrects.double() / dataset_size
    # assert type(final_loss) == np.float, 'final_loss type: {}'.format(type(final_loss))
    # assert type(final_acc) == np.float, 'final_acc type: {}'.format(type(final_acc))
    return {'loss': final_loss, 'acc': final_acc.item()}


def train(model,
          optimizer,
          dataloaders,
          criterion,
          device,
          config,
          num_epochs,
          writer
          ):

    dataset_sizes = config['dataset_size']

    # store train stats
    loss_list = {'train': list(), 'test': list()}
    acc_list = {'train': list(), 'test': list()}
    # iterate over epochs
    for epoch in tqdm.tqdm(range(num_epochs)):
        # learn
        with torch.enable_grad():
            model.train()
            for batch, truth in dataloaders['train']:
                batch = batch.to(device)
                truth = truth.to(device)
                optimizer.zero_grad()

                output = model(batch)
                loss = criterion(output, truth)

                loss.backward()
                optimizer.step()

        # evaluate 
        for phase in ['train', 'test']:

            stats = evaluate_model(model, 
                criterion, 
                dataloaders[phase], 
                device, 
                dataset_sizes[phase]
            )

            loss_list[phase].append(stats['loss'])
            acc_list[phase].append(stats['acc'])
            writer.add_scalar('loss/{}'.format(phase), stats['loss'], epoch)
            writer.add_scalar('accuracy/{}'.format(phase), stats['acc'], epoch)

    return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss': loss_list, 'acc': acc_list}


if __name__ == '__main__':

    # debugging utility
    args = parse_args()
    if args.with_pdb:
        import pdb
        pdb.set_trace()

    # fix randomness
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # cool plotting style
    sns.set_style('darkgrid')
    # sns.set_palette('Set2')

    # directory structure
    log_dir = osp.join(args.run, 'logs')
    ckpt_dir = osp.join(args.run, 'ckpt')
    images_dir = osp.join(args.run, 'images')

    if osp.exists(args.run):
        if args.f:
            shutil.rmtree(args.run)
        else:
            raise Exception('{} already exists'.format(args.run))

    for dirname in [args.run, log_dir, ckpt_dir, images_dir]:
        os.makedirs(dirname)

    # save args
    args_path = osp.join(args.run, 'train_args.pkl')
    with open(args_path, 'w+b') as f:
        pkl.dump(args, f)

    # logging
    writer = SummaryWriter(log_dir=log_dir)

    # get dataset mean, std
    mean, std = get_mean_std(args.dataset)
    config = get_dataset_config(args.dataset)
    train_transform, test_transform = get_dataset_transforms(mean, std, args.augment)
    train_data, test_data = get_dataset(args.dataset, args.dataset_root, train_transform, test_transform)
    
    dataloaders = dict()
    dataloaders['train'] = data.DataLoader(train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    dataloaders['test'] = data.DataLoader(test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    # torch device    
    if args.cuda is None:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.cuda)

    # model
    if hasattr(models, args.model):
        model = getattr(models, args.model)(pretrained=args.pre)
    else:
        model = getattr(networks, args.model)(ch=config['ch'], size=config['size'])

    if hasattr(model, 'fc'):
        model.fc.out_features = config['num_classes']
    elif hasattr(model, 'classifier'):
        model.classifier[-1].out_features = config['num_classes']
    else:
        raise Exception("could not change number of logits")
        
    model = model.to(device)

    # log computational graph and params
    log_graph(writer, model, torch.zeros(args.batch_size, config['ch'], config['size'], config['size'], device=device))
    
    # data parallel
    if args.dp:
        model = nn.DataParallel(model)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2, momentum=args.mom)
        
    # ready to train
    system = train(model,
                   optimizer,
                   dataloaders,
                   criterion,
                   device,
                   config,
                   args.num_epochs,
                   writer
                   )

    # save
    torch.save(system['model'], osp.join(ckpt_dir, 'model_weights.pth'))

    # train stats as npz
    acc_stats_path = osp.join(ckpt_dir, 'acc_stats.npz')
    loss_stats_path = osp.join(ckpt_dir, 'loss_stats.npz')
    np.savez(acc_stats_path, **system['acc'])
    np.savez(loss_stats_path, **system['loss'])

    final_test_loss = system['loss']['test'][-1]
    final_test_acc = system['acc']['test'][-1]

    # args into tensorboard
    log_hparams(writer, args, final_test_loss, final_test_acc)

    acc_image_path = osp.join(images_dir, 'acc.png')
    loss_image_path = osp.join(images_dir, 'loss.png')

    plt.clf()
    plt.figure()
    plt.plot(system['loss']['test'], label='test')
    plt.plot(system['loss']['train'], label='train')
    plt.legend()
    plt.savefig(loss_image_path)

    plt.clf()
    plt.figure()
    plt.plot(system['acc']['test'], label='test')
    plt.plot(system['acc']['train'], label='train')
    plt.legend()
    plt.savefig(acc_image_path)

    print('test acc:', system['acc']['test'][-1])
    print('train acc:', system['acc']['train'][-1])
    writer.close()
