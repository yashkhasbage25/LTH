#1/usr/bin/env python3

import os
import glob
import torch
import argparse
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def parse_args():

    file_purpose = '''
    convert numpy arrays of fake images to png images
    '''

    parser = argparse.ArgumentParser(description=file_purpose,
        epilog=file_purpose,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    default_dpi = 200

    parser.add_argument('-r', '--run', type=str, required=True, help='run dir')
    parser.add_argument('-dpi', type=int, default=default_dpi, help='dpi of image saved')
    parser.add_argument('-pdb', action='store_true', help='run with pdb')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    
    # debugging utils
    if args.pdb:
        import pdb
        pdb.set_trace()

    logs_dir = osp.join(args.run, 'logs')
    images_dir = osp.join(args.run, 'images')
    ckpt_dir = osp.join(args.run, 'ckpt')

    for dirname in [args.run, logs_dir, images_dir, ckpt_dir]:
        assert osp.exists(dirname), '{} was not found'.format(dirname)


    arr_paths = glob.glob(osp.join(ckpt_dir, 'gene_*.pth'), recursive=False)
    for arr_path in arr_paths:
    
        Pm = arr_path[len(osp.join(ckpt_dir, 'gene_')):][:-len('.pth')]
        arrs = torch.load(arr_path)

        epochs = arrs.keys()
        for epoch in epochs:
            plt.close('all')
            plt.axis('off')
            plt.title('{:.2f}%, epoch={}'.format(float(Pm), epoch))
            plt.imshow(np.transpose(vutils.make_grid(arrs[epoch], padding=2, normalize=True).cpu(), (1, 2, 0)))

            image_path = osp.join(images_dir, 'gene_Pm={:.3e}_epoch={}.png'.format(float(Pm), epoch))
            plt.savefig(image_path, dpi=args.dpi)

            print('image saved at:', image_path)