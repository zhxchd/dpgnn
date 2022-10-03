import os
from torch_geometric.datasets import LastFMAsia
from torch_geometric.transforms import NormalizeFeatures

def load_lastfm(root):
    dataset = LastFMAsia(root=os.path.join(root), transform=NormalizeFeatures())
    return dataset

