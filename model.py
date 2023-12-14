import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import degree
import torch_geometric.transforms as T
import torch_geometric.utils as U
import torch_geometric
import networkx as nx
import time
import scipy.sparse as sp

def to_device(X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    return X

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, mode: int = 1,
                 base_model=GCNConv, k: int = 2,cutrate:float=0.2,cutway:int=1,tau:float=0.5):
        super(Encoder, self).__init__()
        self.base_model =base_model
        self.mode = mode
        print("Encoder mode:", self.mode)
        self.tau = tau
        self.cutrate=cutrate
        self.cutway=cutway
        assert k >= 2
        self.k = k

        self.conv = [base_model(in_channels,2*out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, idx: int, pre_z1, pre_z2):
        if pre_z1 == None or (self.mode != 2 and self.mode != 4):
            edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index,edge_weight))
            return x

        pre_z1.detach()
        pre_z2.detach()
        n = pre_z1.shape[0]
        edge_coo = to_scipy_sparse_matrix(edge_index, num_nodes=x.shape[0])
        edge_coo = torch.tensor(edge_coo.toarray(), device=x.device)
        neighbors = edge_coo.sum(1)
        to_device(neighbors)
        neighbors = (neighbors + 1) ** 0.5
        neighbors = neighbors.reshape(pre_z1.shape[0], 1)
        P = None
        if idx == 1:
            P=self.sim(pre_z1,pre_z1)
        if idx == 2:
            P = self.sim(pre_z1, pre_z2)

        P=P.detach()
        P = torch.exp(P / torch.tensor(self.tau))
        P=P/torch.sum(P, dim=1).reshape(n, 1)
        P5k = P.diag().unsqueeze(1).expand(n, n)
        P = P / neighbors
        P = P / neighbors.reshape(1, pre_z1.shape[0])
        P=(P-torch.mean(P))/torch.std(P)
        mask = torch.zeros_like(P).to(x.device)
        edge_index_2=None

        if self.cutway==2:
            flattened_P = P.flatten().to(x.device)
            k = edge_index.shape[1]*self.cutrate
            k=int(k)
            edge_weight, topk_indices = torch.topk(flattened_P, k, largest=True, sorted=True)
            index_i = topk_indices // P.shape[0]
            index_j = topk_indices % P.shape[0]
            edge_index_2 = torch.stack([index_i, index_j])
            mask = to_scipy_sparse_matrix(edge_index_2, num_nodes=x.shape[0])
            mask= torch.tensor(mask.toarray(), device=x.device)

        P12 = torch.sigmoid(P)
        Pf=P12
        if idx == 2:
            W = 1.0 / neighbors
            W = W / neighbors.reshape(1, n)
            W.detach()
            P5k = (P5k - 1) * W
            P5k = abs(P5k)
            P5k = (P5k - torch.mean(P5k)) / torch.std(P5k)
            P5k=torch.sigmoid(P5k)
            Pf = (P5k + P12) / 2.0

        if self.cutway==1:
            Pf = Pf * edge_coo
        else :
            Pf =Pf * mask

        edge_weight = torch.masked_select(Pf, Pf > 0)
        edge_weight.detach()

        if (self.cutway==1 and edge_weight.shape[0] != edge_index.shape[1]  ) \
            or  (self.cutway==2 and edge_weight.shape[0] != edge_index_2.shape[1] ):
            print('err')
            if self.cutway==1:
                edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)
            else :
                edge_weight = torch.ones(edge_index_2.shape[1], dtype=torch.float32, device=x.device)

        if self.cutway==1:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index,edge_weight))
        else:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index_2, edge_weight))

        return x

class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, mode: int = 1,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau
        self.mode = mode
        print("Model mode:", self.mode)
        self.idx = None
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor, idx, pre_z1=None, pre_z2=None) -> torch.Tensor:
        return self.encoder(x, edge_index, idx, pre_z1, pre_z2)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())


    def get_W(self, z1: torch.Tensor, edge_index, type, Pm: torch.Tensor):
        n = z1.shape[0]
        W = torch.zeros(n, n, dtype=torch.float32,device=z1.device)
        edge_coo = to_scipy_sparse_matrix(edge_index, num_nodes=n)
        edge_coo = torch.tensor(edge_coo.toarray(), device=z1.device)
        neighbors = edge_coo.sum(1)
        to_device(neighbors)
        neighbors = (neighbors + 1) ** 0.5
        neighbors = neighbors.reshape(n, 1)
        edge = edge_index.T
        edge.to(z1.device)
        P = Pm.clone()/torch.sum(Pm,1).reshape(n,1)
        P=P.detach()
        P_bef = P.diag().unsqueeze(1).expand(n, n)
        P_bef=P_bef.detach()
        P = P / neighbors
        P = P / neighbors.reshape(1, n)

        if type == 'between':
            P12= (P - torch.mean(P)) / torch.std(P)
            P12=torch.sigmoid(-P12)
            P12 = edge_coo * P12
            # 2
            mask = edge_coo
            sum2 = (mask * P).sum(axis=0).reshape(1, n)
            sum2 = torch.squeeze(sum2)
            # 3
            sum2=sum2.reshape(n,1)
            P3 = P_bef.diag().reshape(n,1)
            P3 = (P3 - 1) / (neighbors ** 2)
            # 2+3
            P23 = sum2 + P3
            P23=torch.diag(torch.squeeze(P23))
            P23 = (P23 - torch.mean(P23)) / torch.std(P23)
            P23 = torch.sigmoid(P23)
            # 4
            mask = torch.mm(mask, mask)
            Pi = P * neighbors
            Pi = Pi * mask
            Pi = Pi / neighbors.reshape(1, n)
            sum4 = Pi.sum(axis=0).reshape(1, n)
            sum4 = torch.squeeze(sum4)
            to_device(sum4)
            # 5k
            W5k = 1.0 / neighbors
            W5k = W5k / neighbors.reshape(1, n)
            P5k = (P_bef - 1) * W5k
            P5k = abs(P5k)
            P5k = (P5k - torch.mean(P5k)) / torch.std(P5k)
            P5k=torch.sigmoid(-P5k)
            P5k=P5k*edge_coo
            # 5
            P5=P5k.sum(axis=0).reshape(1,n)
            # 4+5
            P45=sum4+P5
            P45=torch.diag(torch.squeeze(P45))
            P45 = (P45 - torch.mean(P45)) / torch.std(P45)
            P45=torch.sigmoid(P45)

            W += (P5k + P12) / 2.0
            W += (P23 + P45) / 2.0

        else:
            P12 = (P - torch.mean(P)) / torch.std(P)
            P12=torch.sigmoid(-P12)
            P12=edge_coo*P12
            to_device(P12)
            W+=P12

        W_cpy=torch.clone(W)
        const1=torch.ones(n,n,dtype=torch.float32,device=z1.device)
        W= torch.where(W==0, const1, W_cpy)
        return W

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, edge_index1, edge_index2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        refl_W = torch.ones(z1.shape[0], z1.shape[0])
        between_W = torch.ones(z1.shape[0], z1.shape[0])

        if self.mode == 3 or self.mode == 4:
            refl_W = self.get_W(z1, edge_index1, 'refl', refl_sim)
            between_W = self.get_W(z1, edge_index2, 'between', between_sim)

        refl_W.detach()
        between_W.detach()
        refl_W = to_device(refl_W)
        between_W = to_device(refl_W)

        refl_sim = refl_sim * refl_W
        between_sim = between_sim * between_W

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]
            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, edge_index1: torch.Tensor, edge_index2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):

        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2, edge_index1, edge_index2)
            l2 = self.semi_loss(h2, h1, edge_index1, edge_index2)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
