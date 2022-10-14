from models.gat import GAT
from models.gcn import GCN
from models.graphsage import GraphSAGE
from models.mlp import MLP

def make_model(model_type, hidden_channels, num_features, num_classes, dropout_p=0.5):
    if model_type == "gcn":
        return GCN(hidden_channels=hidden_channels, num_features=num_features, num_classes=num_classes, dropout_p=dropout_p)
    elif model_type == "mlp":
        return MLP(hidden_channels=hidden_channels, num_features=num_features, num_classes=num_classes, dropout_p=dropout_p)
    elif model_type == "graphsage":
        return GraphSAGE(hidden_channels=hidden_channels, num_features=num_features, num_classes=num_classes, dropout_p=dropout_p)
    elif model_type == "gat":
        return GAT(hidden_channels=hidden_channels, num_features=num_features, num_classes=num_classes, dropout_p=dropout_p)