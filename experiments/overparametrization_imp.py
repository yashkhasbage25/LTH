#!/usr/bin/env python3

import os
import sys
import tqdm
import torch
import shutil
import logging
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
import torch.optim.lr_scheduler as lr_scheduler

from torchvision import models

from dataset_utils import *
from lottery_masks import LotteryMask


def parse_args():

    file_purpose = '''
    train a network for lottery tickets
    '''

    parser = argparse.ArgumentParser(description=file_purpose,
        epilog=file_purpose, 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    model_choices = ['FC', 'Conv']
    act_choices = ['ReLU', 'Identity', 'Tanh', 'Sigmoid']

    default_lr = 1e-3
    default_l2 = 0.0
    default_num_epochs = 100
    default_batch_size = 64
    default_n_hidden = 5
    default_hidden_dim = 100
    default_width = 5
    default_act = 'ReLU'
    default_workers = 2
    default_dataset_root = osp.join(osp.dirname(os.getcwd()) ,'datasets')
    default_seed = 0
    default_momentum = 0.9
    default_start = 100.0
    default_end = 1.0
    default_steps = 10
    default_step_gamma = 0.1
    default_milestones = [50, 75]


    parser.add_argument('-lr', type=float, default=default_lr, help='learning rate')
    parser.add_argument('-l2', type=float, default=default_l2, help='l2 penalty')
    parser.add_argument('-n', '--num_epochs', type=int, default=default_num_epochs, help='number of training epochs')
    parser.add_argument('-d', '--dataset', type=str, choices=dataset_choices, required=True, help='dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=default_batch_size, help='batch size for training')
    parser.add_argument('-j', '--workers', type=int, default=default_workers, help='number of wrokers for dataloader')
    parser.add_argument('-m', '--model', type=str, choices=model_choices, required=True, help='model')
    parser.add_argument('-n_hidden', type=int, default=default_n_hidden, help='number of hidden layers (different semantics for FC and Conv')
    parser.add_argument('-hidden_dim', type=int, default=default_hidden_dim, help='dimension of linear layers')
    parser.add_argument('-bn', action='store_true', help='use batch-norm before activations')
    parser.add_argument('-width', type=int, default=default_width, help='width multiplier for Conv model')
    parser.add_argument('-act', type=str, default=default_act, choices=act_choices, help='activation function')
    parser.add_argument('-r', '--run', type=str, required=True, help='run directory prefix')
    parser.add_argument('-f', action='store_true', help='force rewrite')
    parser.add_argument('-dp', action='store_true', help='data parallel model')
    parser.add_argument('-mom', type=float, default=default_momentum, help='momentum for SGD')
    parser.add_argument('-pre', action='store_true', help='pretrained imagenet weights')
    parser.add_argument('-end', type=float, default=default_end, help='end')
    parser.add_argument('-start', type=float, default=default_start, help='start')
    parser.add_argument('-steps', type=int, default=default_steps, help='number of pruning steps')
    parser.add_argument('-cuda', type=int, help='use cuda, if use, then give gpu number')
    parser.add_argument('--seed', type=int, default=default_seed, help='seed for randomness')
    parser.add_argument('--augment', action='store_true', help='augment data with random-flip and random crop')
    parser.add_argument('--milestones', type=int, nargs='+', default=default_milestones, help='milestones for multistep-lr')
    parser.add_argument('--step_gamma', type=float, default=default_step_gamma, help='step gamma for multistep lr')
    parser.add_argument('--dataset_root', type=str, default=default_dataset_root, help='directory for dataset')
    parser.add_argument('-pdb', '--with_pdb', action='store_true', help='run with python debugger')

    return parser.parse_args()



##################
# networks
##################

class FC(nn.Module):

    def __init__(self, n_hidden=10, hidden_dim=10, input_shape=(1, 32, 32), n_classes=10, bn=False, activation='relu'):

        super(FC, self).__init__()
        assert n_hidden >= 2, 'n_hidden has to be >= 2'

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.input_size = 1
        for n in input_shape:
            self.input_size = self.input_size * n
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.bn = bn
        self.activation = activation

        if self.activation in ['ReLU', 'Tanh', 'Sigmoid', 'Identity']:
            self.activation_fn = getattr(nn, self.activation)
        else:
            raise Exception('activation fn not allowed: {}'.format(self.activation))

        layers = [nn.Linear(self.input_size, self.hidden_dim)]
        layers.append(self.activation_fn())
        if self.bn:
            layers.append(nn.BatchNorm1d(self.hidden_dim))

        for layer_index in range(n_hidden - 2):
            layers.append(self.activation_fn())
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            if self.bn:
                layers.append(nn.BatchNorm1d(self.hidden_dim))

        layers.append(self.activation_fn())

        layers.append(nn.Linear(self.hidden_dim, self.n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        out = x.view(x.size(0), self.input_size)
        out = self.net(out)

        return out


class Conv(nn.Module):

    def __init__(self, n_hidden=3, hidden_dim=100, width=5, ch=1, input_shape=(1, 32, 32), n_classes=10, bn=False, activation='relu'):

        super(Conv, self).__init__()
        assert n_hidden >= 2, 'n_hidden has to be  >= 2'

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.width = width
        self.ch = ch
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.bn = bn
        self.activation = activation
        if self.activation in ['ReLU', 'Tanh', 'Sigmoid', 'Identity']:
            self.activation_fn = getattr(nn, self.activation)
        else:
            raise Exception('activation fn not allowed: {}'.format(self.activation))

        width = self.width

        features = list()
        features.append(nn.Conv2d(self.ch, width, kernel_size=3, padding=1))
        features.append(self.activation_fn())
        if self.bn:
            features.append(nn.BatchNorm2d(width))

        for layer_index in range(self.n_hidden - 2):
            features.append(self.activation_fn())
            features.append(nn.Conv2d(width, 2*width, kernel_size=3, padding=1))
            if self.bn:
                features.append(nn.BatchNorm2d(d))
            features.append(nn.MaxPool2d(2, 2))

            width = width * 2

        self.features = nn.Sequential(*features)

        self.features_dim = self.get_features().view(2, -1).size(1)

        classifier = list()
        classifier.append(nn.Linear(self.features_dim, hidden_dim))
        classifier.append(nn.BatchNorm1d(hidden_dim))
        classifier.append(self.activation_fn())
        classifier.append(nn.Linear(hidden_dim, self.n_classes))

        self.classifier = nn.Sequential(*classifier)

    def get_features(self, x=None):

        if x is None:
            input_shape = (2,) + self.input_shape
            x = torch.rand(input_shape)
        
        with torch.no_grad():
            out = self.features(x)

        return out

    def forward(self, x):

        out = self.features(x)
        out = out.view(out.size(0), self.features_dim)
        out = self.classifier(out)

        return out


##################
# train
##################

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
          mask,
          optimizer,
          scheduler,
          dataloaders,
          criterion,
          device,
          config,
          num_epochs,
        #   writer,
          logger,
          pruning_index
          ):

    dataset_sizes = config['dataset_size']

    # store train stats
    loss_list = {'train': list(), 'test': list()}
    acc_list = {'train': list(), 'test': list()}
    # iterate over epochs
    for epoch in range(num_epochs):
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

                mask.apply_mask_to_grads(model)
                optimizer.step()

        scheduler.step()
        # evaluate 
        logger.info('epoch: {}'.format(epoch))
        for phase in ['train', 'test']:
            logger.info('{}:'.format(phase))
            stats = evaluate_model(model, 
                criterion, 
                dataloaders[phase], 
                device, 
                dataset_sizes[phase]
            )

            loss_list[phase].append(stats['loss'])
            acc_list[phase].append(stats['acc'])
            logger.info('\tloss: {}'.format(stats['loss']))
            logger.info('\tacc: {}'.format(stats['acc']))
            # writer.add_scalar('loss-{}/{}'.format(pruning_index, phase), stats['loss'], epoch)
            # writer.add_scalar('accuracy-{}/{}'.format(pruning_index, phase), stats['acc'], epoch)

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
    sns.set_style('whitegrid')
    sns.set_palette('Set2')

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
    args_path = osp.join(args.run, 'train_lottery.pkl')
    with open(args_path, 'w+b') as f:
        pkl.dump(args, f)

    # logging
    # writer = SummaryWriter(log_dir=log_dir)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging_file = osp.join(log_dir, 'train_lottery.log')
    logger = logging.getLogger('train_lottery')
    with open(logging_file, 'w+') as f:
        pass
    logger_file_handler = logging.FileHandler(logging_file)
    logger.addHandler(logger_file_handler)
    logger.info('arguments: {}'.format(args))

    # get dataset mean, std
    mean, std = get_mean_std(args.dataset)
    config = get_dataset_config(args.dataset)
    train_transform, test_transform = get_dataset_transforms(mean, std, config['size'], augment=args.augment)
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
    device = torch.device('cuda:%d' % args.cuda)

    # model
    input_shape = (config['ch'], config['size'], config['size'])
    if args.model == 'FC':
        model = FC(n_hidden=args.n_hidden, 
            hidden_dim=args.hidden_dim, 
            input_shape=input_shape,
            n_classes=config['num_classes'],
            bn=args.bn,
            activation=args.act
        )
    elif  args.model == 'Conv':
        model = Conv(n_hidden=args.n_hidden,
            hidden_dim=args.hidden_dim,
            width=args.width,
            ch=config['ch'],
            input_shape=input_shape,
            n_classes=config['num_classes'],
            bn=args.bn,
            activation=args.act
        )
    else:
        raise Exception('unknown model: {}'.format(args.model))

    # model_weights_path = osp.join(ckpt_dir, 'model_weights.pth')
    # assert osp.exists(model_weights_path), '{} was not found'.format(model_weights_path)
    # model.load_state_dict(torch.load(model_weights_path))
    
    model = model.to(device)

    # log computational graph and params
    # log_graph(writer, model, torch.zeros(args.batch_size, config['ch'], config['size'], config['size'], device=device))
    
    # data parallel
    if args.dp:
        model = nn.DataParallel(model)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # save init
    init_model_weights_path = osp.join(ckpt_dir, 'init_weights.pth')
    torch.save(model.state_dict(), init_model_weights_path)

    # mask of ones
    mask = LotteryMask(model, device, start=args.start, end=args.end, steps=args.steps)

    # start pruning
    for pruning_index in range(args.steps):

        logger.info('pruning_index: {}'.format(pruning_index))

        # optimizer
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2, momentum=args.mom)

        # scheduler
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.step_gamma)
            
        model = mask.apply_mask_to_weights(model)
        # ready to train
        system = train(model,
                    mask,
                    optimizer,
                    scheduler,
                    dataloaders,
                    criterion,
                    device,
                    config,
                    args.num_epochs,
                    # writer,
                    logger,
                    pruning_index
                    )

        unpruned_count, overall_count = mask.get_pruned_stats()
        Pm = unpruned_count / overall_count * 100.0
        print('Pm:', Pm.item(), '%')

        mask.update_mask(model)
        # mask 0 action
        mask.prune_to_zero(model)
        # mask 1 action
        init_state_dict = torch.load(init_model_weights_path)
        mask.reset_to_init(model, init_state_dict)

        # save
        torch.save({
                'model': system['model'],
                'mask': mask.get_mask()
            }, osp.join(ckpt_dir, 'model_weights_{:.3f}.pth'.format(Pm))
        )

        # train stats as npz
        acc_stats_path = osp.join(ckpt_dir, 'train_lottery_acc_stats_{:.3e}.npz'.format(Pm))
        loss_stats_path = osp.join(ckpt_dir, 'train_lottery_loss_stats_{:.3e}.npz'.format(Pm))
        np.savez(acc_stats_path, **system['acc'])
        np.savez(loss_stats_path, **system['loss'])

        # final_test_loss = system['loss']['test'][-1]
        # final_test_acc = system['acc']['test'][-1]

        # args into tensorboard
        # log_hparams(writer, args, final_test_loss, final_test_acc)

        print('test acc:', system['acc']['test'][-1])
        print('train acc:', system['acc']['train'][-1])
    # writer.close()
