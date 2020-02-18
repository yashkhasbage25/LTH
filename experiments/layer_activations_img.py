#!/usr/bin/env python3

import os
import copy
import torch
import argparse
import numpy as np
import pprint as pp
import os.path as osp
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.utils as vutils

import networks

from dataset_utils import *
from lottery_masks import LotteryMask


def parse_args():

    file_purpose = '''
    see layer activations
    '''

    parser = argparse.ArgumentParser(description=file_purpose,
        epilog=file_purpose,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    default_seed = 0
    default_workers = 2
    default_batch_size = 16
    default_cuda = 0
    default_fs = 7
    default_dataset_root = osp.join(osp.dirname(os.getcwd()), 'datasets')
    default_dpi = 200

    parser.add_argument('-m', '--model', type=str, required=True, help='model')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=default_batch_size, help='batch size')
    parser.add_argument('-r', '--run', type=str, required=True, help='run directory')
    parser.add_argument('-l', '--layers', type=str, nargs='+', help='layer name')
    parser.add_argument('-cuda', type=int, default=default_cuda, help='gpu number')
    parser.add_argument('-dp', action='store_true', help='data parallel')
    parser.add_argument('-fs', type=float, default=default_fs, help='title fontsize')
    parser.add_argument('-j', '--workers', type=int, default=default_workers, help='workers')
    parser.add_argument('-ckpts', nargs='+', type=str, required=True, help='name of checkpoints')
    parser.add_argument('-seed', type=int, default=default_seed, help='random seed')
    parser.add_argument('--dataset_root', type=str, default=default_dataset_root, help='dataset root')
    parser.add_argument('-dpi', type=int, default=default_dpi, help='dpi for images')
    parser.add_argument('-pdb', action='store_true', help='run with pdb')

    return parser.parse_args()

def get_layer(model, layer_name):

    for layer_name_, m in model.named_modules():
        if layer_name_ == layer_name:
            if not isinstance(m, (nn.ReLU,)):
                print('layer {} is not ReLU'.format(layer_name))
            return m
        
    raise Exception('layer with name {} was not found'.format(layer_name))

def get_relu_layer_names(model):

    relu_layers = list()
    for layer_name, m in model.named_modules():
        if isinstance(m, nn.ReLU):
            relu_layers.append(layer_name)
        elif isinstance(m, nn.Linear):
            break
    
    if len(relu_layers) == 0:
        raise Exception('model has no relu activations')

    return relu_layers


def get_activations(name):

    def hook(model, x, y):
        activations[name] = y.detach()
    return hook


if __name__ == '__main__':

    args = parse_args()

    # debug
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

    for dirname in [args.run, log_dir, ckpt_dir, images_dir]:
        assert osp.exists(dirname), '{} was not found'.format(dirname)

    ckpt_paths = list()
    for ckpt_name in args.ckpts:
        ckpt_path = osp.join(ckpt_dir, ckpt_name)
        ckpt_paths.append(ckpt_path)
        assert osp.exists(ckpt_path), '{} was not found'.format(ckpt_path)

    # get dataset mean, std
    mean, std = get_mean_std(args.dataset)
    config = get_dataset_config(args.dataset)
    train_transform, test_transform = get_dataset_transforms(mean, std, config['size'], augment=False)
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
    models = [networks.get_model(args.model, args.dataset).to(device) for _ in args.ckpts]

    # data parallel
    if args.dp:
        models = [nn.DataParallel(model) for model in models]
    for i, model in enumerate(models):
        model.load_state_dict(torch.load(ckpt_paths[i])['model'])

    batch = next(iter(dataloaders['train']))[0].to(device)
    # print('layer descriptions:')
    # pp.pprint([(name, str(type(m))) for name, m in models[0].named_modules()])

    if args.layers is None:
        layers = get_relu_layer_names(models[0])
    else:
        layers = args.layers

    activations = dict()


    for layer_name in layers:
        for model_index, model in enumerate(models):
            layer = get_layer(model, layer_name)
            layer.register_forward_hook(get_activations('model{}:{}'.format(model_index, layer_name)))

    for model in models:
        model(batch)

    x = batch

    figsize = (2 * len(models) + 2, 2  * len(layers))
    fig, axs = plt.subplots(ncols=(len(models) + 1), nrows=len(layers), figsize=figsize)
    axs = axs.reshape((len(layers), len(models) + 1))

    for layer_index, layer_name in enumerate(layers):
        for model_index, model in enumerate(models):

            activation_key = 'model{}:{}'.format(model_index, layer_name)
            mean_activation = activations[activation_key].mean(2).mean(2)
            max_val, max_index = torch.max(mean_activation, 1)
            max_activations = torch.zeros_like(activations[activation_key][:, 0])
            for sample_index in range(args.batch_size):
                max_activations[sample_index] = activations[activation_key][sample_index][max_index[sample_index]]
            
            x_img = vutils.make_grid(x, nrow=4, padding=1, normalize=True).detach().cpu().numpy().transpose(1, 2, 0)
            if max_activations.dim() == 3:
                max_activations = max_activations.unsqueeze(1)

            activations_img = vutils.make_grid(max_activations, nrow=4, padding=1, normalize=True)
            activations_img = activations_img.detach().cpu().numpy().transpose(1, 2, 0)
            axs[layer_index][0].imshow(x_img)
            axs[layer_index][model_index + 1].imshow(activations_img)

    for layer_index in range(len(layers)):
        for model_index in range(len(models) + 1):
            axs[layer_index][model_index].set_xticks([])
            axs[layer_index][model_index].set_yticks([])
    
    for layer_index, layer in enumerate(layers):
        axs[layer_index][0].set_ylabel(layer)

    Pm_flt_list = [float(ckpt[len('final_weights_Pm_'):][:-len('.pth')]) for ckpt in args.ckpts]

    axs[0][0].set_title('images', fontsize=args.fs)
    for model_index in range(len(args.ckpts)):
        axs[0][model_index + 1].set_title('Pm={:.3f}'.format(Pm_flt_list[model_index]), fontsize=args.fs)

    fig.tight_layout()
    image_path = osp.join(images_dir, 'layer_activations.png')
    fig.savefig(image_path, dpi=args.dpi)
    print('image saved at:', image_path)
