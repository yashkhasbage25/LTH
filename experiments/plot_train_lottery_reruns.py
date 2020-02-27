#!/usr/bin/env python3

import os
import glob
import argparse
import numpy as np
import pickle as pkl
import os.path as osp
import seaborn as sns
import matplotlib.pyplot as plt

from dataset_utils import *


def parse_args():

    file_purpose = '''
    plot reruns stats for lottery tickets
    '''

    parser = argparse.ArgumentParser(description=file_purpose,
        epilog=file_purpose,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    default_dpi = 200

    parser.add_argument('-r', '--run', type=str, required=True, help='run dir')
    parser.add_argument('-dpi', type=int, default=default_dpi, help='image dpi')
    parser.add_argument('-pdb', action='store_true', help='run with pdb')

    return parser.parse_args()

def get_pruning_ratios(train_args):

    start = train_args.start
    end = train_args.end
    steps = train_args.steps

    common_ratio = np.power((end / start), 1.0 / (steps - 1) )

    pm = list()
    ratio = start
    for pruning_index in range(steps):
        pm.append(ratio)
        ratio = ratio * common_ratio
    
    pm_flt = pm
    pm_str = ['{:.3e}'.format(pm_) for pm_ in pm]

    return pm_flt, pm_str


if __name__ == '__main__':

    args = parse_args()

    # debugging utils
    if args.pdb:
        import pdb
        pdb.set_trace()

    # directory structure
    ckpt_dir, images_dir, log_dir = get_directories(args.run)

    for dirname in [args.run, ckpt_dir, images_dir, log_dir]:
        assert osp.exists(dirname), '{} was not found'.format(dirname)

    train_args_path = osp.join(args.run, 'train_lottery.pkl')
    with open(train_args_path, 'rb') as f:
        train_args = pkl.load(f)

    pm_flt, pm_str = get_pruning_ratios(train_args)

    acc_stats = list()
    
    for rerun_index in range(train_args.reruns):
        acc_stats.append(list())
        for pm in pm_str:
            stats_path = osp.join(ckpt_dir, 'train_lottery_acc_stats_rerun_{}_{}.npz'.format(rerun_index, pm))
            acc_stats[-1].append(np.load(stats_path)['test'][-1])

    acc_stats = np.array(acc_stats)

    assert acc_stats.shape == (args.reruns, len(pm_str)), 'shape mismath: {} != {}'.format(acc_stats.shape, (args.reruns, len(pm_str)))

    mean = np.mean(acc_stats, axis=0)
    std = np.mean(acc_stats, axis=0)

    plt.plot(mean)
    plt.fill_between(np.arange(len(pm_str)), mean-std, mean+std)

    plt.xlabel('weights_remaining')
    plt.ylabel('accuracy')
    
    plt.xticks(np.arange(len(pm_str)), pm_str)

    image_path = osp.join(images_dir, 'train_lottery_reruns.png')
    plt.savefig(image_path, dpi=args.dpi)

    print('image saved at:', image_path)