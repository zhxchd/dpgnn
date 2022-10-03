import os
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def load_pubmed(root):
    dataset = Planetoid(root=os.path.join(root, "Planetoid"), name='pubmed', transform=NormalizeFeatures())
    return dataset

