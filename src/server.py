import math
import numpy as np
import torch
from models import make_model

class Server:
    def __init__(self, eps, delta, data) -> None:
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
        self.priv_adj = priv_adj
        self.priv_deg = priv_deg
        # project priv_deg to [1, n-1], otherwise resulting in useless prior = 0 or 1
        self.priv_deg[priv_deg < 1] = 1
        self.priv_deg[priv_deg > self.n - 1] = self.n - 1

    def estimate(self):

        def estimate_prior():
            def phi(x):
                r = 1.0/(torch.exp(x).matmul(torch.ones(1,self.n)) + torch.ones(self.n,1).matmul(torch.exp(-x).reshape(1,self.n)))
                return torch.log(self.priv_deg) - torch.log(r.sum(1).reshape(self.n,1) - r.diagonal().reshape(self.n,1))
            
            beta = torch.zeros(self.n, 1)

            # beta is a fixed point for phi
            for _ in range(500):
                beta = phi(beta)

            s = torch.ones(self.n, 1).matmul(beta.transpose(0,1)) + beta.matmul(torch.ones(1,self.n))
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
        dth_max = pij.sort(dim=1, descending=True).values.gather(1, self.priv_deg.long() - 1)
        self.est_edge_index = (pij >= dth_max).float().to_sparse().coalesce().indices()
        # self.est_edge_index = (pij > 0.5).float().to_sparse().coalesce().indices()
        # self.est_edge_index = torch.bernoulli(pij).to_sparse().coalesce().indices()

    def fit(self, model, d, c, hparam, iter=200):
        log = np.zeros((iter, 3))

        # we train the model on GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = make_model(model_type=model, hidden_channels=16, num_features=d, num_classes=c).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hparam["lr"], weight_decay=hparam["wd"])
        criterion = torch.nn.CrossEntropyLoss()

        self.data = self.data.to(device)
        if self.priv:
            self.est_edge_index = self.est_edge_index.to(device)
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