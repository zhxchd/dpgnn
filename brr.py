import torch
import numpy as np
from torch_sparse import SparseTensor

class BRR():
    def __init__(self, eps, num_nodes, edge_index):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.p = 1.0/(1.0 + torch.exp(torch.tensor(eps)).to(self.device))
        self.n = num_nodes
        self.adj = SparseTensor(
            row=edge_index[0], col=edge_index[1], sparse_sizes=(self.n, self.n)).to(self.device).to_dense()
        self.adj_t = self.adj.transpose(0, 1)

    def __actual_degree(self):
        return self.adj.sum(1).reshape(self.n, 1)

    def __rr(self):
        # return 1 with probability p, but does not flip diagonal edges since no self loop allowed
        return ((self.adj + torch.bernoulli(torch.full((self.n, self.n), self.p).to(self.device))) % 2) * (1 - torch.eye(self.n, self.n).to(self.device))

    def __get_pij_2d(self, noisy_adj, deg):
        amb = deg.matmul(deg.transpose(0, 1))
        # apb = torch.ones(n,1).to(device).matmul(deg.transpose(0,1)) + deg.matmul(torch.ones(1,n).to(device))
        # prior = amb/(2*amb + n*n - apb*(n-1) - 2*n+1) * (1 - torch.eye(n, n).to(device)) # given the degree list, the probability that there's an edge between node i and j, this is our prior
        prior = amb / (deg.sum() - 1) * (1 - torch.eye(self.n, self.n).to(self.device))
        prior[prior > 1] = 1
        noisy_adj_t = noisy_adj.transpose(0, 1)
        x = noisy_adj + noisy_adj_t
        # x can be 0 (0,0), 1 (1, 0) / (0, 1), 2 (1,1)
        pr_y_edge = 0.5*(x-1)*(x-2)*self.p*self.p + 0.5*x*(x-1) * (1-self.p)*(1-self.p) - 1*x*(x-2)*self.p*(1-self.p)
        pr_y_no_edge = 0.5*(x-1)*(x-2)*(1-self.p)*(1-self.p) + 0.5 *x*(x-1)*self.p*self.p - 1*x*(x-2)*self.p*(1-self.p)
        pij = pr_y_edge * prior / (pr_y_edge * prior + pr_y_no_edge * (1 - prior))
        return pij

    def get_est_edge_index(self):
        noisy_adj = self.__rr()
        deg = self.__actual_degree()
        pij = self.__get_pij_2d(noisy_adj, deg)
        
        pij = ((pij >= 0.5) + (pij.transpose(0, 1) >= 0.5)).float()
        
        sparse_pij = pij.to_sparse()
        est_edge_index = sparse_pij.indices()
        return est_edge_index
