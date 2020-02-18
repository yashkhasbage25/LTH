#!/usr/bin/env python3

import os
import glob
import torch
import argparse
import numpy as np
import seaborn as sns
import os.path as osp
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def parse_args():

    file_purpose = '''
    comparing gan images from two different lottery tickets
    '''
    parser = argparse.ArgumentParser(description=file_purpose,
        epilog=file_purpose,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    default_dpi = 400
    default_ls = 3
    default_nrows = 10

    parser.add_argument('-r', '--run', type=str, required=True, help='run dir')
    parser.add_argument('-nrows', type=int, default=default_nrows, help='number of rows in each image')
    parser.add_argument('-d', '--diff', type=int, required=True, help='difference b/w epochs')
    parser.add_argument('-ls', type=float, default=default_ls, help='label size')
    parser.add_argument('-dpi', type=int, default=default_dpi, help='dpi for image')
    parser.add_argument('-pdb', action='store_true', help='run with pdb')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    # debugging
    if args.pdb:
        import pdb
        pdb.set_trace()

    # cool plotting
    # sns.set_palette('Set2')
    # sns.set_style('white')

    # directory structure
    log_dir = osp.join(args.run, 'logs')
    images_dir = osp.join(args.run, 'images')
    ckpt_dir = osp.join(args.run, 'ckpt')

    for dirname in [args.run, log_dir, images_dir, ckpt_dir]:
        assert osp.exists(dirname), '{} was not found'.format(dirname)

    # see all stored image paths and extract Pm
    images_path = glob.glob(osp.join(ckpt_dir, 'gene_*.pth'), recursive=False)
    Pm_str_list = [path[len(osp.join(ckpt_dir, 'gene_')):][:-len('.pth')] for path in images_path]
    Pm_flt_list = [float(Pm) for Pm in Pm_str_list]
    Pm_flt_list = sorted(Pm_flt_list)

    # for each Pm get images
    images = dict()
    for Pm in Pm_flt_list:
        
        images_path = osp.join(ckpt_dir, 'gene_{:.3e}.pth'.format(Pm))
        images['{:.3e}'.format(Pm)] = torch.load(images_path)

    max_epoch = max(images[next(iter(images.keys()))].keys())

    nrows = len(Pm_flt_list)
    ncols = max_epoch // args.diff

    figsize = (6 * nrows, 6 * ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

    axs = axs.reshape((nrows, ncols))

    for row, Pm in enumerate(Pm_flt_list):
        for col in range(0, ncols):

            epoch = col * args.diff

            epoch_images = images['{:.3e}'.format(Pm)][epoch]
            epoch_images = vutils.make_grid(epoch_images.cpu(), nrow=args.nrows, padding=1, normalize=True)
            axs[row][col].imshow(np.transpose(epoch_images, (1, 2, 0)))
            axs[row][col].set_xticks([])
            axs[row][col].set_yticks([])

    for i in range(ncols):
        axs[0][i].set_title('epoch {}'.format(i * args.diff), fontsize=args.ls)
    
    for i, Pm in enumerate(Pm_flt_list):
        axs[i][0].set_ylabel('Pm={:.3f}'.format(Pm), fontsize=args.ls)

    fig.tight_layout(pad=0.5)

    image_path = osp.join(images_dir, 'compare_evolution.png')
    fig.savefig(image_path, dpi=args.dpi)
    print('image saved at:', image_path)