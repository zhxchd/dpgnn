import math
import numpy as np
import torch
from models import make_model
from torchmetrics.functional.classification import multiclass_f1_score

class Server:
    def __init__(self, eps, delta, data) -> None:
        # no privacy
        if eps == None:
            self.priv = False
        else:
            self.priv = True
            if delta == None:
                # do not privatize degree sequence
                self.priv_deg = False
                self.eps_a = eps
                self.eps_d = None
            else:
                self.priv_deg = True
                self.eps_d = eps * delta
                self.eps_a = eps * (1-delta)
        self.data = data
        self.n = data.num_nodes

    def receive(self, priv_adj, priv_deg):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.priv_adj = priv_adj.to(device)
        self.priv_deg = priv_deg.to(device)
        # project priv_deg to [1, n-2], otherwise resulting in useless prior = 0 or 1
        self.priv_deg[priv_deg < 1] = 1
        self.priv_deg[priv_deg > self.n - 2] = self.n - 2

    def estimate(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # store 1 vectors to save RAM
        ones_1xn = torch.ones(1,self.n).to(device)
        ones_nx1 = torch.ones(self.n,1).to(device)

        def estimate_prior():
            def phi(x):
                r = 1.0/(torch.exp(x).matmul(ones_1xn) + ones_nx1.matmul(torch.exp(-x).reshape(1,self.n)))
                return torch.log(self.priv_deg) - torch.log(r.sum(1).reshape(self.n,1) - r.diagonal().reshape(self.n,1))
            
            beta = torch.zeros(self.n, 1).to(device)

            # beta is a fixed point for phi
            for _ in range(200):
                beta = phi(beta)

            s = ones_nx1.matmul(beta.transpose(0,1)) + beta.matmul(ones_1xn)
            prior = torch.exp(s)/(1+torch.exp(s))
            prior.fill_diagonal_(0)
            return prior

        def estimate_posterior(prior):
            p = 1.0/(1.0+math.exp(self.eps_a))
            priv_adj_t = self.priv_adj.transpose(0,1)
            x = self.priv_adj + priv_adj_t
            pr_y_edge = 0.5*(x-1)*(x-2)*p*p + 0.5*x*(x-1)*(1-p)*(1-p) - 1*x*(x-2)*p*(1-p)
            pr_y_no_edge = 0.5*(x-1)*(x-2)*(1-p)*(1-p) + 0.5*x*(x-1)*p*p - 1*x*(x-2)*p*(1-p)
            pij = pr_y_edge * prior / (pr_y_edge * prior + pr_y_no_edge * (1 - prior))
            return pij
        
        pij = estimate_posterior(estimate_prior())
        # this is to choose the top degree edges of each node, and also add weights
        # dth_max = pij.sort(dim=1, descending=True).values.gather(1, self.priv_deg.long() - 1)
        # weighted_edges = (pij * (pij >= dth_max)).float().to_sparse().coalesce()
        # self.est_edge_index = weighted_edges.indices()
        # self.est_edge_value = weighted_edges.values()

        # hard threshold of 0.5
        self.est_edge_index = (pij >= 0.5).float().to_sparse().coalesce().indices()
        ones_1xn = None # reset variable so that it's no longer used and VRAM can be freed
        ones_nx1 = None
        # take random graph based on pij
        # self.est_edge_index = torch.bernoulli(pij).to_sparse().coalesce().indices()

    def fit(self, model, hparam, iter=200):
        log = np.zeros((iter, 4))

        # we train the model on GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = make_model(model_type=model, hidden_channels=16, num_features=self.data.num_features, num_classes=self.data.num_classes, dropout_p=hparam["do"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hparam["lr"], weight_decay=hparam["wd"])
        criterion = torch.nn.CrossEntropyLoss()

        self.data = self.data.to(device)
        if self.priv:
            self.est_edge_index = self.est_edge_index.to(device)
            # self.est_edge_value = self.est_edge_value.to(device)
        else:
            self.est_edge_index = self.data.edge_index

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
            pred = out.argmax(dim=1)
            f1 = multiclass_f1_score(pred[self.data.val_mask], self.data.y[self.data.val_mask], num_classes=self.data.num_classes)
            return float(loss), float(f1)
        
        def test():
            model.eval()
            out = model(self.data.x, self.est_edge_index)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            f1 = multiclass_f1_score(pred[self.data.test_mask], self.data.y[self.data.test_mask], num_classes=self.data.num_classes)
            return float(f1)
        
        for epoch in range(1, iter+1):
            train_loss = train()
            val_loss, val_f1 = validate()
            test_f1 = test()
            log[epoch-1] = [train_loss, val_loss, val_f1, test_f1]
        
        # return numpy array of iter rows,
        # each row is (train loss, validation loss, validation f1 score, and test f1 score)
        return log