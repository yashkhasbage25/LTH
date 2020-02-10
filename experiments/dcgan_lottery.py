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

import networks

from dataset_utils import *
from lottery_masks import LotteryMask

def parse_args():

    file_purpose = '''
    dcgan for cifar
    '''

    parser = argparse.ArgumentParser(description=file_purpose,
        epilog=file_purpose,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    default_lr = 1e-4
    default_seed = 0
    default_workers = 2
    default_num_epochs = 20
    default_z_len = 100
    default_batch_size = 128
    default_cuda = 0
    default_start = 100.0
    default_end = 1.0
    default_steps = 10
    default_step_gamma = 0.1
    default_milestones = [10, 20]
    default_dataset_root = osp.join(osp.dirname(os.getcwd()), 'datasets')

    parser.add_argument('-lr', type=float, default=default_lr, help='learning rate')
    parser.add_argument('-n', '--num_epochs', type=int, default=default_num_epochs, help='epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=default_batch_size, help='batch size')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='dataset')
    parser.add_argument('-cuda', type=int, default=default_cuda, help='gpu number')
    parser.add_argument('-seed', type=int, default=default_seed, help='random seed')
    parser.add_argument('-z_len', type=int, default=default_z_len, help='len of noise vector')
    parser.add_argument('-f', action='store_true', help='force rewrite')
    parser.add_argument('-end', type=float, default=default_end, help='end')
    parser.add_argument('-start', type=float, default=default_start, help='start')
    parser.add_argument('-steps', type=int, default=default_steps, help='number of pruning steps')
    parser.add_argument('-r', '--run', type=str, required=True, help='run directory prefix')
    parser.add_argument('--milestones', type=int, nargs='+', default=default_milestones, help='milestones for multistep lr')
    parser.add_argument('--step_gamma', type=float, default=default_step_gamma, help='gamma for multistep lr')
    parser.add_argument('--augment', action='store_true', help='augment data')
    parser.add_argument('-j', '--workers', type=int, default=default_workers, help='workers')
    parser.add_argument('-dp', action='store_true', help='data parallel')
    parser.add_argument('--dataset_root', type=str, default=default_dataset_root, help='dataset root')
    parser.add_argument('-pdb', action='store_true', help='run with debugger')

    return parser.parse_args()


def generate_images(G, z_len, device, nsamples=100):

    with torch.no_grad():
        G.eval()
        z = torch.randn(nsamples, z_len, device=device).view(nsamples, z_len, 1, 1)
        images = G(z)

    return images.detach().cpu()

def train(D, 
          G,
          dmask,
          gmask,
          doptim,
          goptim,
          dscheduler,
          gscheduler,
          dataloader,
          criterion,
          z_len,
          device,
          config,
          num_epochs,
          logger,
          pruning_index
          ):

    train_dataset_size = config['dataset_size']
    images_per_epoch = dict()
    
    # store train stats
    gloss_list = list()
    dloss_list = list()

    # iterate over epochs
    for epoch in range(num_epochs):

        D.train()
        G.train()

        # learn 
        for batch, _ in dataloader:

            # reset grads to zero
            # gtopim.zero_grad()
            doptim.zero_grad()

            # form data
            batch = batch.to(device)

            batch_size = batch.size(0)
            y_real = torch.ones(batch_size).to(device)
            y_fake = torch.zeros(batch_size).to(device)

            # discriminator step
            d_prob = D(batch)
            d_prob = d_prob.squeeze()
            d_real_loss = criterion(d_prob, y_real)

            z = torch.randn(batch_size, z_len).view(batch_size, z_len, 1, 1).to(device)
            fake_img = G(z)

            d_prob = D(fake_img).squeeze()
            d_fake_loss = criterion(d_prob, y_fake)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            doptim.step()

            dloss_list.append(d_loss.item())

            # generator step
            goptim.zero_grad()

            z = torch.randn(batch_size, z_len).view(batch_size, z_len, 1, 1).to(device)
            fake_image = G(z)
            d_prob = D(fake_image).squeeze()
            g_loss = criterion(d_prob, y_real)
            g_loss.backward()
            goptim.step()

            gloss_list.append(g_loss.item())

        # scheduler
        gscheduler.step()
        dscheduler.step()

        # log
        # dloss_list.append(dloss)
        # gloss_list.append(g_loss)
        logger.info('epoch: {}'.format(epoch))
        logger.info('\tgloss: {}'.format(gloss_list[-1]))
        logger.info('\tdloss: {}'.format(dloss_list[-1]))

        images = generate_images(G, z_len, device, nsamples=100)
        images_per_epoch[epoch] = images

    return {
                'D': D.state_dict(),
                'G': G.state_dict(),
                'doptim': doptim.state_dict(),
                'goptim': goptim.state_dict(),
                'gloss': np.array(gloss_list),
                'dloss': np.array(dloss_list),
                'images_per_epoch': images_per_epoch
            }

if __name__ == '__main__':

    # args
    args = parse_args()
    
    # debugger 
    if args.pdb:
        import pdb
        pdb.set_trace()

    # fix randomness
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # cool plotting
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
    args_path = osp.join(args.run, 'train_args.pkl')
    with open(args_path, 'w+b') as f:
        pkl.dump(args, f)

    # logging 
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
    
    dataloader = data.DataLoader(train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    ) 

    # torch device    
    device = torch.device('cuda:%d' % args.cuda)

    D, G = networks.get_model('dcgan', args.dataset, utility='gan')

    G = G.to(device)
    D = D.to(device)

    # data parallel
    if args.dp:
        generator = nn.DataParallel(generator)
        discriminator =  nn.DataParallel(discriminator)

    # loss function
    criterion = nn.BCELoss()

    # save init
    init_G_weights_path = osp.join(ckpt_dir, 'init_generator.pth')
    init_D_weights_path = osp.join(ckpt_dir, 'init_discriminator.pth')

    torch.save(G.state_dict(), init_G_weights_path)
    torch.save(D.state_dict(), init_D_weights_path)

    # mask of ones
    gmask = LotteryMask(G, device, start=args.start, end=args.end, steps=args.steps)
    dmask = LotteryMask(D, device, start=args.start, end=args.end, steps=args.steps)

    # start pruning
    for pruning_index in range(args.steps):

        logger.info('pruning_index: {}'.format(pruning_index))

        # optimizer
        goptim = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
        doptim = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

        # scheduler
        gscheduler = lr_scheduler.MultiStepLR(goptim, args.milestones, gamma=args.step_gamma)
        dscheduler = lr_scheduler.MultiStepLR(doptim, args.milestones, gamma=args.step_gamma)

        G = gmask.apply_mask_to_weights(G)
        D = dmask.apply_mask_to_weights(D)

        # train
        system = train(D, G, 
                    dmask, gmask, 
                    doptim, goptim, 
                    dscheduler, gscheduler, 
                    dataloader,
                    criterion, 
                    args.z_len,
                    device, 
                    config, 
                    args.num_epochs, 
                    logger, 
                    pruning_index,
                )

        unpruned_count, overall_count = dmask.get_pruned_stats()
        Pm = unpruned_count * 100.0 / overall_count
        print('Pm:', Pm.item(), '%')

        # images = generate_images(G, samples=100)
        images_path = osp.join(ckpt_dir, 'gene_{:.3e}.pth'.format(Pm.item()))
        # np.savez(images_path, **system['images_per_epoch'])
        torch.save(system['images_per_epoch'], images_path)

        # pruning criterion
        dmask.update_mask(D)
        gmask.update_mask(G)

        # mask 0 action
        dmask.prune_to_zero(D)
        gmask.prune_to_zero(G)

        # mask 1 action
        dmask.reset_to_init(D, torch.load(init_D_weights_path))
        gmask.reset_to_init(G, torch.load(init_G_weights_path))

        gloss_stats_path = osp.join(ckpt_dir, 'train_lottery_gloss_stats_{:.3e}.npy'.format(Pm))
        dloss_stats_path = osp.join(ckpt_dir, 'train_lottery_dloss_stats_{:.3e}.npy'.format(Pm))
        np.save(gloss_stats_path, system['gloss'])
        np.save(dloss_stats_path, system['dloss'])