import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (
    coalesce,
    negative_sampling,
    structured_negative_sampling,
)
from conv import FirstSNEAconv, NoFirstSNEAconv
from typing import Optional, Tuple, Any, Union
from torch_geometric.nn.conv import MessagePassing
from layers import *

class MySNEA(nn.Module):
    def __init__(self, in_dims, out_dims, layers, params) -> None:
        super().__init__()
        hid_dims = out_dims // 2
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.first_aggr = FirstSNEAconv(in_dims, hid_dims, g_sigma=params.g_sigma, c=params.c, ablation=params.ablation)
        self.layers = layers
        self.aggrs = nn.ModuleList()
        for i in range(self.layers - 1):
            self.aggrs.append(NoFirstSNEAconv(hid_dims, hid_dims, g_sigma=params.g_sigma, c=params.c, ablation=params.ablation))
        self.W = nn.Parameter(torch.zeros(out_dims, out_dims))

        self.lamb = 4
        self.lp_lin = nn.Linear(2 * out_dims, 3)
        self.reset_parameters()

    def reset_parameters(self):
        self.first_aggr.reset_parameters()

        for i in range(self.layers - 1):
            self.aggrs[i].reset_parameters()

        self.lp_lin.reset_parameters()
        torch.nn.init.xavier_normal_(self.W)

    def forward(self, x, pos_edge_index, neg_edge_index, head):
        pos_x, neg_x, pos_m, neg_m = self.first_aggr(x, x, pos_edge_index, neg_edge_index, head)
        for i in range(self.layers - 1):
            pos_x, neg_x, pos_m, neg_m = self.aggrs[i](pos_x, neg_x, pos_edge_index, neg_edge_index, head)

        return F.tanh(torch.cat((pos_x, neg_x), dim=1) @ self.W), torch.cat([pos_m,neg_m], dim=-1)

    def discriminate(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """Given node embeddings :obj:`z`, classifies the link relation
        between node pairs :obj:`edge_index` to be either positive,
        negative or non-existent.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
        """
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lp_lin(value)
        return torch.log_softmax(value, dim=1)

    def link_prediction(self, z: Tensor, edge_index: Tensor) -> Tensor:
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lp_lin(value)[:, :2]
        return F.softmax(value, dim=1)

    def test(
            self,
            z: Tensor,
            pos_edge_index: Tensor,
            neg_edge_index: Tensor,
    ) -> Tuple[float, float]:
        from sklearn.metrics import f1_score, roc_auc_score

        with torch.no_grad():
            pos_p = self.discriminate(z, pos_edge_index)[:, :2].max(dim=1)[1]
            neg_p = self.discriminate(z, neg_edge_index)[:, :2].max(dim=1)[1]
        pred = (1 - torch.cat([pos_p, neg_p])).cpu()
        y = torch.cat(
            [pred.new_ones(pos_p.size(0)),
             pred.new_zeros(neg_p.size(0))])
        pred, y = pred.numpy(), y.numpy()

        auc = roc_auc_score(y, pred)
        f1 = f1_score(y, pred, average='binary') if pred.sum() > 0 else 0
        return auc, f1

    def nll_loss(
            self,
            z: Tensor,
            pos_edge_index: Tensor,
            neg_edge_index: Tensor,
    ) -> Tensor:
        """Computes the discriminator loss based on node embeddings :obj:`z`,
        and positive edges :obj:`pos_edge_index` and negative nedges
        :obj:`neg_edge_index`.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))

        nll_loss = 0
        nll_loss += F.nll_loss(
            self.discriminate(z, pos_edge_index),
            pos_edge_index.new_full((pos_edge_index.size(1),), 0))
        nll_loss += F.nll_loss(
            self.discriminate(z, neg_edge_index),
            neg_edge_index.new_full((neg_edge_index.size(1),), 1))
        nll_loss += F.nll_loss(
            self.discriminate(z, none_edge_index),
            none_edge_index.new_full((none_edge_index.size(1),), 2))
        return nll_loss / 3.0

    def pos_embedding_loss(
            self,
            z: Tensor,
            pos_edge_index: Tensor,
    ) -> Tensor:
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
        """
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))

        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z: Tensor, neg_edge_index: Tensor) -> Tensor:
        """Computes the triplet loss between negative node pairs and sampled
        non-node pairs.

        Args:
            z (torch.Tensor): The node embeddings.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))

        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def loss(
            self,
            z: Tensor,
            pos_edge_index: Tensor,
            neg_edge_index: Tensor,
    ) -> Tensor:
        """Computes the overall objective.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_1 = self.pos_embedding_loss(z, pos_edge_index)
        loss_2 = self.neg_embedding_loss(z, neg_edge_index)
        return nll_loss + self.lamb * (loss_1 + loss_2)

    def create_spectral_features(
            self,
            pos_edge_index: Tensor,
            neg_edge_index: Tensor,
            num_nodes: Optional[int] = None,
    ) -> Tensor:
        import scipy.sparse as sp
        from sklearn.decomposition import TruncatedSVD

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        N = edge_index.max().item() + 2 if num_nodes is None else num_nodes
        edge_index = edge_index.to(torch.device('cpu'))

        pos_val = torch.full((pos_edge_index.size(1),), 2, dtype=torch.float)
        neg_val = torch.full((neg_edge_index.size(1),), 0, dtype=torch.float)
        val = torch.cat([pos_val, neg_val], dim=0)

        row, col = edge_index
        edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
        val = torch.cat([val, val], dim=0)

        edge_index, val = coalesce(edge_index, val, num_nodes=N)
        val = val - 1

        # Borrowed from:
        # https://github.com/benedekrozemberczki/SGCN/blob/master/src/utils.py
        edge_index = edge_index.detach().numpy()
        val = val.detach().numpy()
        A = sp.coo_matrix((val, edge_index), shape=(N, N))
        svd = TruncatedSVD(n_components=self.in_dims, n_iter=128)
        svd.fit(A)
        x = svd.components_.T
        return torch.from_numpy(x).to(torch.float).to(pos_edge_index.device)

    def compute_tri_edge_accuracy(
            self,
            degrees: torch.Tensor,
            pos_edge_index: torch.Tensor,
            neg_edge_index: torch.Tensor,
            z: Tensor,
    ) -> tuple[Union[Union[float, int], Any], Union[Union[float, int], Any]]:
        pos_p, neg_p = self.test_pos_eng(z, pos_edge_index, neg_edge_index)
        multi_correct_predictions = 0
        single_correct_predictions = 0
        m = 0
        s = 0
        k1 = 70
        k2 = 30
        for i in range(pos_edge_index.size(1)):
            src, dst = pos_edge_index[:, i]
            if pos_p[i] == 0:
                if degrees[src] >= k1 and degrees[dst] >= k1:
                    multi_correct_predictions = multi_correct_predictions + 1
                elif degrees[src] <= k2 and degrees[dst] <= k2:
                    single_correct_predictions = single_correct_predictions + 1
                    s = s + 1
            else:
                if degrees[src] >= k1 and degrees[dst] >= k1:
                    m = m + 1
                elif degrees[src] <= k2 and degrees[dst] <= k2:
                    s = s + 1
        for i in range(neg_edge_index.size(1)):
            src, dst = neg_edge_index[:, i]
            if neg_p[i] == 1:
                if degrees[src] >= k1 and degrees[dst] >= k1:
                    multi_correct_predictions = multi_correct_predictions + 1
                    m = m + 1
                elif degrees[src] <= k2 and degrees[dst] <= k2:
                    single_correct_predictions = single_correct_predictions + 1
                    s = s + 1
            else:
                if degrees[src] >= k1 and degrees[dst] >= k1:
                    m = m + 1
                elif degrees[src] <= k2 and degrees[dst] <= k2:
                    s = s + 1
        if m != 0:
            multi_acc = multi_correct_predictions / m
        else:
            multi_acc = 0
        if s != 0:
            single_acc = single_correct_predictions / s
        else:
            single_acc = 0
        return multi_acc, single_acc

    def test_pos_eng(
            self,
            z: Tensor,
            pos_edge_index: Tensor,
            neg_edge_index: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            pos_p = self.discriminate(z, pos_edge_index)[:, :2].max(dim=1)[1]
            neg_p = self.discriminate(z, neg_edge_index)[:, :2].max(dim=1)[1]
        return pos_p, neg_p