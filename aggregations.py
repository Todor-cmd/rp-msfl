import torch


def fedmes_adjustment(gradients, overlap_weight_index):
    params_to_add = []
    for client_id in range(len(gradients)):
        if overlap_weight_index[client_id] == 2:
            params_to_add.append(gradients[client_id])
        elif overlap_weight_index[client_id] == 3:
            params_to_add.append(gradients[client_id])
            params_to_add.append(gradients[client_id])

    return gradients + params_to_add


def fedmes_median(gradients, overlap_weight_index):
    adjusted_gradients = fedmes_adjustment(gradients, overlap_weight_index)
    return torch.median(adjusted_gradients, dim=0)[0]


def fedmes_mean(gradients, overlap_weight_index):
    adjusted_gradients = fedmes_adjustment(gradients, overlap_weight_index)
    return torch.mean(adjusted_gradients, dim=0)[0]
