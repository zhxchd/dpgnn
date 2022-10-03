from torch_geometric.data import dataset
from data.cora import load_cora
from data.lastfm import load_lastfm
from data.pubmed import load_pubmed

def make_dataset(dataset_name, root):
    if dataset_name == "cora":
        return load_cora(root)
    elif dataset_name == "pubmed":
        return load_pubmed(root)
    elif dataset_name == "lastfm":
        return load_lastfm(root)