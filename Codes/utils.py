
import torch
import torch.nn.functional as F
import numpy as np

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)


def hard2chandep(chan_deps, params = None, device = None):
    if device is None:
        device = torch.device("cpu")
    split_deps = torch.split(chan_deps, 1, 1)        
    split_deps = list(split_deps)
    b_sz, __, h_sz, w_sz = split_deps[0].size()
    alpha = torch.sigmoid(split_deps[2])
    max_dep =  (alpha > 0.5).long()
    max_dep = max_dep.permute(0, 2, 3, 1).view(-1, 1)
    dep_cat = torch.cat((F.relu(split_deps[1]), F.relu(split_deps[0])), 1)*params['depth_maxrange']
    dep_cat = dep_cat.permute(0, 2, 3, 1).view(-1, 2)
    final_dep = torch.gather(dep_cat, 1, max_dep)
    final_dep = final_dep.view(b_sz, h_sz, w_sz, 1).permute(0, 3, 1, 2)

    return final_dep

def smooth2chandep(chan_deps, params = None, device = None):
    if device is None:
        device = torch.device("cpu")
    split_deps = torch.split(chan_deps, 1, 1)        
    split_deps = list(split_deps)

    alpha = torch.sigmoid(split_deps[2])
    final_dep = alpha*F.relu(split_deps[0])*params['depth_maxrange'] + (1 - alpha)*F.relu(split_deps[1])*params['depth_maxrange']

    return final_dep

