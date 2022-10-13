import os
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def load_citeseer(root):
    dataset = Planetoid(root=os.path.join(root, "Planetoid"), name='CiteSeer', transform=NormalizeFeatures(), split="random", num_train_per_class=300, num_val=832, num_test=832)
    return dataset