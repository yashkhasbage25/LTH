#!/usr/bin/env python3

import glob
import argparse
import numpy as np
import pickle as pkl
import os.path as osp
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():

    file_purpose = '''
    plotting script for train_lottery.py
    '''

    parser = argparse.ArgumentParser(description=file_purpose,
        epilog=file_purpose,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    default_dpi = 100

    parser.add_argument('-r', '--run', type=str, required=True, help='run dir')
    parser.add_argument('-dpi', type=int, default=default_dpi, help='dpi for image')
    parser.add_argument('-pdb', action='store_true', help='run with pdb')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    if args.pdb:
        import pdb
        pdb.set_trace()

    # sns.set_style('darkgrid')
    sns.set_style('whitegrid')
    sns.set_palette('Set2')

    ckpt_dir = osp.join(args.run, 'ckpt')
    images_dir = osp.join(args.run, 'images')
    acc_stats_path = osp.join(ckpt_dir, 'train_lottery_acc_stats_*.npz')
    acc_stats_paths = glob.glob(acc_stats_path)

    Pm_list = [float(acc_stats_path[len(osp.join(ckpt_dir, 'train_lottery_acc_stats_')):][:-len('.npz')]) for acc_stats_path in acc_stats_paths]
    Pm_list = reversed(sorted(Pm_list))

    x_ticks = list()
    acc_list = list()
    
    for Pm in Pm_list:

        acc_stats_path = osp.join(ckpt_dir, 'train_lottery_acc_stats_{:.3e}.npz'.format(Pm))
        assert osp.exists(acc_stats_path), '{} was not found'.format(acc_stats_path)
        stats = np.load(acc_stats_path)
        x_ticks.append(Pm)
        acc_list.append(stats['test'][-1])

    fig = plt.figure()
    plt.plot(acc_list)
    plt.plot([acc_list[0]] * len(acc_list), label='unpruned')
    plt.xticks(np.arange(len(x_ticks)), ['{:.3f}'.format(x) for x in x_ticks])

    image_path = osp.join(images_dir, 'train_lottery.png')
    fig.savefig(image_path, dpi=args.dpi)

    print('image saved at:', image_path)