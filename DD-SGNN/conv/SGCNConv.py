import torch
import torch.nn.functional as F
from layers import *
from typing import Union
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, PairTensor, SparseTensor
from torch_geometric.utils import spmm

class mySignedConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, first_aggr: bool,
                 g_sigma, c, bias: bool = True, ablation=0, **kwargs):

        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_aggr = first_aggr
        self.ablation = ablation
        self.g = Generator(in_channels, g_sigma, ablation)
        self.c = c

        if first_aggr:
            self.lin_pos_l = Linear(in_channels, out_channels, False)
            self.lin_pos_r = Linear(in_channels, out_channels, bias)
            self.lin_neg_l = Linear(in_channels, out_channels, False)
            self.lin_neg_r = Linear(in_channels, out_channels, bias)
            self.lin_m = Linear(in_channels, 2 * out_channels, False)
            self.r = Relation(in_channels, ablation)
        else:
            self.lin_pos_l = Linear(2 * in_channels, out_channels, False)
            self.lin_pos_r = Linear(in_channels, out_channels, bias)
            self.lin_neg_l = Linear(2 * in_channels, out_channels, False)
            self.lin_neg_r = Linear(in_channels, out_channels, bias)
            self.lin_m = Linear(in_channels, out_channels, False)
            self.r = Relation(2 * in_channels, ablation)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_pos_l.reset_parameters()
        self.lin_pos_r.reset_parameters()
        self.lin_neg_l.reset_parameters()
        self.lin_neg_r.reset_parameters()
        self.lin_m.reset_parameters()

    def forward(
            self,
            x: Union[Tensor, PairTensor],
            pos_edge_index: Adj,
            neg_edge_index: Adj,
            head
    ):
        '''pos_mean = F.normalize(pos_adj, p=1, dim=1)
        pos_neighbor = torch.mm(pos_mean, x)
        neg_mean = F.normalize(neg_adj, p=1, dim=1)
        neg_neighbor = torch.mm(neg_mean, x)
        pos_m = self.r(x, pos_neighbor)
        neg_m = self.r(x, neg_neighbor)'''

        if isinstance(x, Tensor):
            x = (x, x)
        pos_neighbor = self.propagate(pos_edge_index, x=x)
        pos_m = self.r(x[0], pos_neighbor)
        neg_neighbor = self.propagate(pos_edge_index, x=x)
        neg_m = self.r(x[0], neg_neighbor)

        # propagate_type: (x: PairTensor)
        if self.first_aggr:
            if head:
                out_pos = self.propagate(pos_edge_index, x=x)
                out_pos = self.lin_pos_l(out_pos)
                out_pos = out_pos + self.lin_pos_r(x[1])

                out_neg = self.propagate(neg_edge_index, x=x)
                out_neg = self.lin_neg_l(out_neg)
                out_neg = out_neg + self.lin_neg_r(x[1])

            else:
                pos_h_s = pos_m
                neg_h_s = neg_m

                out_pos = self.propagate(pos_edge_index, x=x)
                out_pos = out_pos + self.c * pos_h_s
                out_pos = self.lin_pos_l(out_pos)
                out_pos = out_pos + self.lin_pos_r(x[1])

                out_neg = self.propagate(neg_edge_index, x=x)
                out_neg = out_neg + self.c * neg_h_s
                out_neg = self.lin_neg_l(out_neg)
                out_neg = out_neg + self.lin_neg_r(x[1])

            return torch.cat([out_pos,out_neg], dim=-1), torch.cat([pos_m,neg_m], dim=-1)

        else:
            F_in = self.in_channels
            if head:
                out_pos1 = self.propagate(pos_edge_index,
                                      x=(x[0][..., :F_in], x[1][..., :F_in]))
                out_pos2 = self.propagate(neg_edge_index,
                                      x=(x[0][..., F_in:], x[1][..., F_in:]))
                out_pos = torch.cat([out_pos1, out_pos2], dim=-1)
                out_pos = self.lin_pos_l(out_pos)
                out_pos = out_pos + self.lin_pos_r(x[1][..., :F_in])

                out_neg1 = self.propagate(pos_edge_index,
                                      x=(x[0][..., F_in:], x[1][..., F_in:]))
                out_neg2 = self.propagate(neg_edge_index,
                                      x=(x[0][..., :F_in], x[1][..., :F_in]))
                out_neg = torch.cat([out_neg1, out_neg2], dim=-1)
                out_neg = self.lin_neg_l(out_neg)
                out_neg = out_neg + self.lin_neg_r(x[1][..., F_in:])

            else:
                pos_h_s = pos_m
                neg_h_s = neg_m

                out_pos1 = self.propagate(pos_edge_index,
                                          x=(x[0][..., :F_in], x[1][..., :F_in]))
                out_pos2 = self.propagate(neg_edge_index,
                                          x=(x[0][..., F_in:], x[1][..., F_in:]))

                out_pos = torch.cat([out_pos1, out_pos2], dim=-1)
                out_pos = out_pos + self.c * pos_h_s
                out_pos = self.lin_pos_l(out_pos)
                out_pos = out_pos + self.lin_pos_r(x[1][..., :F_in])

                out_neg1 = self.propagate(pos_edge_index,
                                          x=(x[0][..., F_in:], x[1][..., F_in:]))
                out_neg2 = self.propagate(neg_edge_index,
                                          x=(x[0][..., :F_in], x[1][..., :F_in]))
                out_neg = torch.cat([out_neg1, out_neg2], dim=-1)
                out_neg = out_neg + self.c * neg_h_s
                out_neg = self.lin_neg_l(out_neg)
                out_neg = out_neg + self.lin_neg_r(x[1][..., F_in:])

            return torch.cat([out_pos, out_neg], dim=-1), torch.cat([pos_m,neg_m], dim=-1)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: PairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, first_aggr={self.first_aggr})')