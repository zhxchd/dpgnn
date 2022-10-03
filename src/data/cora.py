import os
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# cora already has its train/val/test split defined, no need for further processing
def load_cora(root):
    dataset = Planetoid(root=os.path.join(root, "Planetoid"), name='Cora', transform=NormalizeFeatures(), split="random", num_train_per_class=200, num_val=677, num_test=677)
    return dataset