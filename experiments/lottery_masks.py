#!/usr/bin/env python3

import torch
import numpy as np

class LotteryMask():

    def __init__(self, model, start=100.0, end=1.0, steps=20):

        self.mask = [torch.ones_like(p) for p in model.parameters()]
        self.common_ratio = np.power((end / start), 1.0/ (steps - 1))

        self.p_m = start * self.common_ratio

        self.layer_indices = dict()
        cumul_sum = 0
        for layer_name, p in model.named_parameters():
            self.layer_indices[layer_name] = (cumul_sum, cumul_sum + torch.numel(p))
            cumul_sum += torch.numel(p)

    def apply_mask_to_weights(self, model):

        with torch.no_grad():
            for (p, m) in zip(model.parameters(), self.mask):
                p.data.copy_(p * m)

        return model

    def apply_mask_to_grads(self, model):

        with torch.no_grad():
            for (p, m) in zip(model.parameters(), self.mask):
                p.grad = p.grad * m

    
    def update_mask(self, model):

        with torch.no_grad():
            new_p_m = self.p_m - self.p_m * self.common_ratio
            pruned_indices = [p == 0.0 for p in model.parameters()]

            weights = torch.cat([p.view(-1) for p in model.parameters()])
            weights = weights.detach().cpu().numpy()
            percentile = np.percentile(abs(weights), (100 - new_p_m))
            updated_mask = np.where(abs(weights) < percentile, 0.0, 1.0)
            self.mask = list()
            for layer_name, p in model.named_parameters():
                lb, ub = self.layer_indices[layer_name]
                self.mask.append(updated_mask[lb:ub])
            self.mask = [torch.from_numpy(m).to(model.device) for m in self.mask]
        # updated_mask = list()
        # for p in model.parameters():
        #     layer_weights = p.data.cpu().numpy()
        #     percentile = np.percentile(abs(layer_weights), (100 - new_p_m))
        #     updated_mask.append(np.where(abs(layer_weights) < percentile, 0.0, 1.0))
            
        # self.mask = [torch.from_numpy(u).to(p.device).float() for u in updated_mask]
        self.p_m = new_p_m

    def prune_to_zero(self, model):

        with torch.no_grad():
            for (p, m) in zip(model.parameters(), self.mask):
                p.data = p.data * m

    def get_pruned_stats(self):

        with torch.no_grad():
            total_params = sum([torch.numel(m) for m in self.mask])
            non_zero = sum([torch.sum(m) for m in self.mask])

        return (non_zero, total_params)

    def reset_to_init(self, model, init_state_dict):
        
        with torch.no_grad():
            for (p_now, p_init, m) in zip(model.parameters(), init_state_dict.values(), self.mask):
                indices = m > 0.0
                p_now[indices] = p_init[indices]
                