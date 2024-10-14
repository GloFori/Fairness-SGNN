from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, softmax, remove_self_loops
from layers import *

__all__ = ['FirstSNEAconv', 'NoFirstSNEAconv']

class FirstSNEAconv(MessagePassing):
    def __init__(self, in_dims, hid_dims, g_sigma, c, ablation, aggr: Union[str, List[str], Aggregation, None] = "add", *, aggr_kwargs:
    Optional[Dict[str, Any]] = None, flow: str = "source_to_target", node_dim: int = -2, decomposed_layers: int = 1, **kwargs):
        super().__init__(aggr, aggr_kwargs=aggr_kwargs, flow=flow,
                         node_dim=node_dim, decomposed_layers=decomposed_layers, **kwargs)

        self.W_pos = nn.Parameter(torch.zeros((in_dims, hid_dims)))
        self.b_pos = nn.Parameter(torch.zeros((1, 2 * hid_dims)))
        self.W_neg = nn.Parameter(torch.zeros((in_dims, hid_dims)))
        self.b_neg = nn.Parameter(torch.zeros((1, 2 * hid_dims)))
        self.ablation = ablation
        self.g = Generator(in_dims, g_sigma, ablation)
        self.c = c
        self.r = Relation(in_dims//2, ablation)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        nn.init.xavier_normal_(self.W_pos)
        nn.init.xavier_normal_(self.W_neg)

    def message(self, x_j: Tensor, alpha, m) -> Tensor:
        if m:
            alpha = None
            return x_j
        else:
            return alpha.view((-1, 1)) * x_j

    def update(self, inputs: Tensor) -> Tensor:
        return F.tanh(inputs)

    def forward(self, pos_x: Tensor, neg_x: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor, head) -> Any:
        pos_edge_index = add_self_loops(
            remove_self_loops(pos_edge_index)[0])[0]
        neg_edge_index = add_self_loops(
            remove_self_loops(neg_edge_index)[0])[0]

        ppos_x = pos_x @ self.W_pos
        nneg_x = neg_x @ self.W_neg

        pos_neighbor = self.propagate(pos_edge_index, x=ppos_x, alpha=None, m=True)
        pos_m = self.r(ppos_x, pos_neighbor)
        neg_neighbor = self.propagate(neg_edge_index, x=nneg_x, alpha=None, m=True)
        neg_m = self.r(nneg_x, neg_neighbor)

        pos_e = F.tanh(self.b_pos @ torch.cat(
            [ppos_x[pos_edge_index[0]], ppos_x[pos_edge_index[1]]], dim=1).T).squeeze()
        neg_e = F.tanh(self.b_neg @ torch.cat(
            [nneg_x[neg_edge_index[0]], nneg_x[neg_edge_index[1]]], dim=1).T).squeeze()

        pos_a = softmax(pos_e, pos_edge_index[0])
        neg_a = softmax(neg_e, neg_edge_index[0])
        if head:
            pos_x = self.propagate(pos_edge_index, x=ppos_x, alpha=pos_a, m=False)
            neg_x = self.propagate(neg_edge_index, x=nneg_x, alpha=neg_a, m=False)
        else:
            pos_x = self.propagate(pos_edge_index, x=ppos_x, alpha=pos_a, m=False)
            pos_x = self.c * pos_m + pos_x
            neg_x = self.propagate(neg_edge_index, x=nneg_x, alpha=neg_a, m=False)
            neg_x = self.c * neg_m + neg_x

        return pos_x, neg_x, pos_m, neg_m


class NoFirstSNEAconv(MessagePassing):
    def __init__(self, in_dims, hid_dims, g_sigma, c, ablation, aggr: Union[str, List[str], Aggregation, None] = "add", *, aggr_kwargs:
    Optional[Dict[str, Any]] = None, flow: str = "source_to_target", node_dim: int = -2, decomposed_layers: int = 1, **kwargs):
        super().__init__(aggr, aggr_kwargs=aggr_kwargs, flow=flow,
                         node_dim=node_dim, decomposed_layers=decomposed_layers, **kwargs)

        self.W_pos = nn.Parameter(torch.zeros((in_dims, hid_dims)))
        self.b_pos = nn.Parameter(torch.zeros((1, 2 * hid_dims)))
        self.W_neg = nn.Parameter(torch.zeros((in_dims, hid_dims)))
        self.b_neg = nn.Parameter(torch.zeros((1, 2 * hid_dims)))
        self.ablation = ablation
        self.g = Generator(in_dims, g_sigma, ablation)
        self.c = c
        self.r = Relation(in_dims, ablation)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        nn.init.xavier_normal_(self.W_pos)
        nn.init.xavier_normal_(self.W_neg)

    def message(self, x_j: Tensor, alpha ,m) -> Tensor:
        if m:
            alpha = None
            return x_j
        else:
            return alpha.view((-1, 1)) * x_j

    def forward(self, pos_x: Tensor, neg_x: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor, head) -> Any:
        pos_neighbor = self.propagate(pos_edge_index, x=pos_x, alpha=None, m=True)
        pos_m = self.r(pos_x, pos_neighbor)
        neg_neighbor = self.propagate(neg_edge_index, x=neg_x, alpha=None, m=True)
        neg_m = self.r(neg_x, neg_neighbor)

        pos_edge_index = add_self_loops(
            remove_self_loops(pos_edge_index)[0])[0]
        neg_edge_index = neg_edge_index

        ppos_x = pos_x @ self.W_pos
        pneg_x = neg_x @ self.W_pos
        nneg_x = neg_x @ self.W_neg
        npos_x = pos_x @ self.W_neg

        ppos_e = F.tanh(
            self.b_pos @ torch.cat([ppos_x[pos_edge_index[0]], ppos_x[pos_edge_index[1]]], dim=1).T).squeeze()
        pneg_e = F.tanh(
            self.b_pos @ torch.cat([pneg_x[neg_edge_index[0]], pneg_x[neg_edge_index[1]]], dim=1).T).squeeze()
        nneg_e = F.tanh(
            self.b_neg @ torch.cat([nneg_x[pos_edge_index[0]], nneg_x[pos_edge_index[0]]], dim=1).T).squeeze()
        npos_e = F.tanh(
            self.b_neg @ torch.cat([npos_x[neg_edge_index[0]], npos_x[neg_edge_index[0]]], dim=1).T).squeeze()

        pos_e = torch.cat([ppos_e, pneg_e], dim=0)
        neg_e = torch.cat([nneg_e, npos_e], dim=0)
        edge_index = torch.cat([pos_edge_index[0], neg_edge_index[0]], dim=0)
        pos_a = softmax(pos_e, edge_index)
        neg_a = softmax(neg_e, edge_index)

        ppos_a = pos_a[:ppos_e.size(0)]
        pneg_a = pos_a[ppos_e.size(0):]
        nneg_a = neg_a[:nneg_e.size(0)]
        npos_a = neg_a[nneg_e.size(0):]
        if head:
            ppos_x = self.propagate(pos_edge_index, x=ppos_x, alpha=ppos_a, m=False)
            pneg_x = self.propagate(neg_edge_index, x=pneg_x, alpha=pneg_a, m=False)
            nneg_x = self.propagate(pos_edge_index, x=nneg_x, alpha=nneg_a, m=False)
            npos_x = self.propagate(neg_edge_index, x=npos_x, alpha=npos_a, m=False)

            return F.tanh(ppos_x + pneg_x), F.tanh(nneg_x + npos_x), pos_m, neg_m
        else:
            ppos_x = self.propagate(pos_edge_index, x=ppos_x, alpha=ppos_a, m=False)
            pneg_x = self.propagate(neg_edge_index, x=pneg_x, alpha=pneg_a, m=False)
            nneg_x = self.propagate(pos_edge_index, x=nneg_x, alpha=nneg_a, m=False)
            npos_x = self.propagate(neg_edge_index, x=npos_x, alpha=npos_a, m=False)

            return F.tanh(ppos_x + pneg_x) + self.c * pos_m, F.tanh(nneg_x + npos_x) + self.c * neg_m, pos_m, neg_m