import math
from typing import Tuple

import torch
from torch_sparse import SparseTensor

class Client():
    def __init__(self, eps, delta, data) -> None:
        self.data = data
        if delta == None:
            # do not privatize degree sequence
            self.priv_deg = False
            self.eps_a = eps
            self.eps_d = None
        else:
            self.priv_deg = True
            self.eps_d = eps * delta
            self.eps_a = eps * (1-delta)

    def AddLDP(self) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.data.num_nodes
        adj = SparseTensor(row=self.data.edge_index[0].cpu(), col=self.data.edge_index[1].cpu(), sparse_sizes=(n, n)).to_dense()
        deg = adj.sum(1).reshape(n, 1)

        def rr_adj() -> torch.Tensor:
            p = 1.0/(1.0+math.exp(self.eps_a))
            # return 1 with probability p, but does not flip diagonal edges since no self loop allowed
            res = ((adj + torch.bernoulli(torch.full((n, n), p))) % 2).float()
            res.fill_diagonal_(0)
            return res

        def laplace_deg() -> torch.Tensor:
            return deg + torch.distributions.laplace.Laplace(loc=0, scale=1/self.eps_d).sample((n,1))

        if self.priv_deg:
            return rr_adj(), laplace_deg()
        else:
            return rr_adj(), deg

    