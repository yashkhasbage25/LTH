#!/usr/bin/env python3

import torch.optim as optim

def get_optimizer(model, args):
    optim_name = args.optimizer
    
    if optim_name == 'Adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif optim_name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif optim_name == 'RMSProp':
        return optim.RMSProp(model.parameters(), lr=args.lr, weight_decay=args.l2, momentum=args.mom)
    elif optim_name == 'SGD':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2, momentum=args.mom)
    else:
        raise Exception('unknown optimizer: {}'.format(optim_name))
