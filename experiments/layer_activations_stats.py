#!/usr/bin/env python3

import os
import copy
import glob
import torch
import argparse
import numpy as np
import pprint as pp
import torch.nn as nn 
import os.path as osp
import seaborn as sns
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.utils as vutils

import networks

from dataset_utils import *
from lottery_masks import *

def parse_args():

    file_purpose = '''
    compare layer activations for non-LTH and LTH model
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
    parser.add_argument('-cuda', type=int, default=default_cuda, help='gpu number')
    parser.add_argument('-relu_only', action='store_true', help='see relu activations only')
    parser.add_argument('-dp', action='store_true', help='data parallel')
    parser.add_argument('-fs', type=float, default=default_fs, help='title fontsize')
    parser.add_argument('-j', '--workers', type=int, default=default_workers, help='workers')
    parser.add_argument('-seed', type=int, default=default_seed, help='random seed')
    parser.add_argument('--dataset_root', type=str, default=default_dataset_root, help='dataset root')
    parser.add_argument('-dpi', type=int, default=default_dpi, help='dpi for images')
    parser.add_argument('-pdb', action='store_true', help='run with pdb')

    return parser.parse_args()


def get_valid_module_names(model, relu_only=True):

    if relu_only:
        valid_nn_modules = nn.ReLU
    else:
        valid_nn_modules = (
            nn.Conv2d, nn.ConvTranspose2d,
            nn.MaxPool2d, nn.MaxUnpool2d,
            nn.AvgPool2d, nn.AdaptiveMaxPool2d,
            nn.AdaptiveAvgPool2d, nn.ELU, 
            nn.LeakyReLU, nn.PReLU, 
            nn.ReLU, nn.Sigmoid, 
            nn.Tanh, nn.Softmax, 
            nn.BatchNorm2d, nn.Linear, 
            nn.Dropout2d 
        )

    valid_names = list()
    for name, module in model.named_modules():
        if isinstance(module, valid_nn_modules):
            valid_names.append(name)

    if len(valid_names) == 0:
        raise Exception('no layer found in model')

    return valid_names

def get_activations(layer_name):

    def hook(model, x, y):
        activations[layer_name] = y.detach()
    
    return hook

def load_state_dict(model, ckpt_path):

    try:
        model.load_state_dict(torch.load(ckpt_path)['model'])
    except RuntimeError:
        model.module.load_state_dict(torch.load(ckpt_path)['model'])
    return model

if __name__ == '__main__':

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
    sns.set_style('darkgrid')
    # sns.set_palette('Set2')

    # directory structure
    log_dir = osp.join(args.run, 'logs')
    ckpt_dir = osp.join(args.run, 'ckpt')
    images_dir = osp.join(args.run, 'images')

    for dirname in [args.run, log_dir, ckpt_dir, images_dir]:
        assert osp.exists(dirname), '{} was not found'.format(dirname)

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
    model = networks.get_model(args.model, args.dataset).to(device)
    model.load_state_dict(torch.load(osp.join(ckpt_dir, 'final_weights_Pm_{}.pth'.format(100.0)))['model'])

    if args.dp:
        model = nn.DataParallel(model)

    # get some data samples
    batch = next(iter(dataloaders['train']))[0].to(device)

    # pp.pprint([name for name, _ in model.named_modules()])

    valid_names = get_valid_module_names(model, relu_only=args.relu_only)
    for module_name, module in model.named_modules():
        if module_name in valid_names:
            module.register_forward_hook(get_activations(module_name))

    activations = dict()
    y = model(batch)
    orig_activations = dict()

    for layer_name, actv in activations.items():
        orig_activations[layer_name] = actv.detach().cpu().numpy()

    ckpt_paths = glob.glob(osp.join(ckpt_dir, 'final_weights_Pm_*.pth'))
    Pm_str_list = [path[len(osp.join(ckpt_dir, 'final_weights_Pm_')):][:-len('.pth')] for path in ckpt_paths]
    
    Pm_flt_list = sorted([float(Pm) for Pm in Pm_str_list])
    Pm_flt_list = Pm_flt_list[:-1]

    figsize = (10, 5)
    fig, axs = plt.subplots(ncols=2, figsize=figsize)

    axs[0].set_title('mean abs diff')
    axs[1].set_title('mean diff')

    for Pm in Pm_flt_list:

        # model = networks.get_model(args.model, args.dataset).to(device)
        ckpt_path = osp.join(ckpt_dir, 'final_weights_Pm_{}.pth'.format(Pm))
        model = load_state_dict(model, ckpt_path)
        # model.load_state_dict(torch.load(ckpt_path)['model'])

        activations = dict()
        pruned_activations = dict()
        y = model(batch)

        for layer_name, actv in activations.items():
            pruned_activations[layer_name] = actv.detach().cpu().numpy()

        mean_abs_diff = list()
        mean_diff = list()

        for layer_name in valid_names:
            mean_diff.append(np.mean(pruned_activations[layer_name] - orig_activations[layer_name]))
            mean_abs_diff.append(np.mean(np.abs(pruned_activations[layer_name] - orig_activations[layer_name])))

        axs[0].plot(mean_abs_diff)
        axs[1].plot(mean_diff, label='{:.3f}%'.format(Pm))
    
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[1].legend(bbox_to_anchor=(1.04, 0.9))

    fig.tight_layout()

    if args.relu_only:
        image_path = osp.join(images_dir, 'layer_activations_diff_relu.png')
    else:
        image_path = osp.join(images_dir, 'layer_activations_diff.png')
    fig.savefig(image_path, dpi=args.dpi, bbox_inches='tight')
    print('image saved at:', image_path)