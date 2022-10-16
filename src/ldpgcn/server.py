import math
import numpy as np
import torch
from models import make_model

class Server:
    def __init__(self, eps, data) -> None:
        self.eps = eps
        self.data = data
        self.n = data.num_nodes

    def receive(self, priv_adj):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.priv_adj = priv_adj.to(device)

    def estimate(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # number of edges
        e = torch.round(self.priv_adj.sum()/2).to(device)
        # upper triangular matrix, consider both a_ij and a_ji
        atr = torch.triu(self.priv_adj, 1) + torch.triu(self.priv_adj.transpose(0,1), 1)
        # get the e largest entris in atr
        e_th = torch.topk(atr.flatten(), e).values.min()
        atr[atr >= e_th] = 1
        atr[atr < e_th] = 0
        self.est_edge_index = (atr + atr.transpose(0,1)).to_sparse().coalesce().indices()

    def fit(self, model, hparam, iter=200):
        log = np.zeros((iter, 3))

        # we train the model on GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = make_model(model_type=model, hidden_channels=16, num_features=self.data.num_features, num_classes=self.data.num_classes, dropout_p=hparam["do"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hparam["lr"], weight_decay=hparam["wd"])
        criterion = torch.nn.CrossEntropyLoss()

        self.data = self.data.to(device)
        self.est_edge_index = self.est_edge_index.to(device)

        def train():
            model.train()
            optimizer.zero_grad()
            out = model(self.data.x, self.est_edge_index)
            loss = criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()
            return float(loss)
        
        def validate():
            model.eval()
            out = model(self.data.x, self.est_edge_index)
            loss = criterion(out[self.data.val_mask], self.data.y[self.data.val_mask])
            return float(loss)
        
        def test():
            model.eval()
            out = model(self.data.x, self.est_edge_index)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            test_correct = pred[self.data.test_mask] == self.data.y[self.data.test_mask]
            test_acc = int(test_correct.sum()) / int(self.data.test_mask.sum())
            return test_acc
        
        for epoch in range(1, iter+1):
            loss = train()
            val_loss = validate()
            test_acc = test()
            log[epoch-1] = [loss, val_loss, test_acc]
        
        return log