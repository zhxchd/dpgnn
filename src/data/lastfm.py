import os
import torch
from torch_geometric.datasets import LastFMAsia
from torch_geometric.transforms import NormalizeFeatures


def load_lastfm(root):
    dataset = LastFMAsia(root=os.path.join(root, "LastFMAsia"), transform=NormalizeFeatures())

    
    # train_mask = torch.full([7624], False, dtype=bool)
    # val_mask = torch.full([7624], False, dtype=bool)
    # test_mask = torch.full([7624], False, dtype=bool)
    # perm = torch.randperm(7624)
    # train_mask[perm[0:3812]] = True
    # val_mask[perm[3812:5718]] = True
    # test_mask[perm[5718:7624]] = True
    # dataset[0].train_mask = train_mask
    # dataset[0].val_mask = val_mask
    # dataset[0].test_mask = test_mask
    return dataset

