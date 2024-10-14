import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (
    coalesce,
    negative_sampling,
)
from typing import Optional, Tuple
from torch_geometric import seed_everything
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import argparse
from layers import *
from conv import GCNConv

# cuda / mps / cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class MySGCL(nn.Module):
    def __init__(self, in_channels, out_channels, layer_num, params) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        #self.args = args
        self.params = params
        # transform
        self.transform = nn.Linear(4 * out_channels, out_channels)

        # predictor
        self.predictor = Predictor(out_channels).to(device)

        self.activation = nn.ReLU()

        # dimension reduce embeddings
        self.linear_DR = nn.Linear(in_channels, out_channels).to(device)

    def dimension_reduction(self):
        return self.linear_DR(self.params.x)

    def drop_edges(self, edge_index, ratio=0.8):
        assert (0 <= ratio and ratio <= 1)
        M = edge_index.size(1)
        tM = int(M * ratio)
        permutation = torch.randperm(M)
        return edge_index[:, permutation[:tM]], edge_index[:, permutation[tM:]]

    def connectivity_perturbation(self, N, pos_edge_index, neg_edge_index, ratio=0.1):
        pos_tM = int(pos_edge_index.size(1) * ratio)
        res_pos_edge_index, _ = self.drop_edges(pos_edge_index, 1 - ratio)
        neg_tM = int(pos_edge_index.size(1) * ratio)
        res_neg_edge_index, _ = self.drop_edges(neg_edge_index, 1 - ratio)

        res_edge_index = torch.cat((res_pos_edge_index, res_neg_edge_index), dim=1)
        sample = negative_sampling(res_edge_index, N, pos_tM + neg_tM)
        pos_edge_index = torch.cat((res_pos_edge_index, sample[:, :pos_tM]), dim=1)
        neg_edge_index = torch.cat((res_neg_edge_index, sample[:, pos_tM:]), dim=1)
        return pos_edge_index, neg_edge_index

    def sign_perturbation(self, N, pos_edge_index, neg_edge_index, ratio=0.1):
        pos_edge_index, to_neg_edge_index = self.drop_edges(pos_edge_index, 1 - ratio)
        neg_edge_index, to_pos_edge_index = self.drop_edges(neg_edge_index, 1 - ratio)

        pos_edge_index = torch.cat((pos_edge_index, to_pos_edge_index), dim=1)
        neg_edge_index = torch.cat((neg_edge_index, to_neg_edge_index), dim=1)
        return pos_edge_index, neg_edge_index

    def generate_view(self, N, pos_edge_index, neg_edge_index):
        con_pos_edge_index, con_neg_edge_index = self.connectivity_perturbation(N, pos_edge_index, neg_edge_index,
                                                                                self.params.aug_ratio)
        sig_pos_edge_index, sig_neg_edge_index = self.sign_perturbation(N, pos_edge_index, neg_edge_index,
                                                                        self.params.aug_ratio)
        return con_pos_edge_index, con_neg_edge_index, sig_pos_edge_index, sig_neg_edge_index

    def encode(self, edge_index_a, edge_index_b, x, head):
        x_a, x_b = None, None

        for _ in range(self.layer_num):
            # encoder = GATConv(self.in_channels, self.out_channels).to(device)
            encoder = GCNConv(self.in_channels, self.out_channels, g_sigma=self.params.g_sigma, c=self.params.c,
                              ablation=self.params.ablation).to(device)

            x_a, m1 = encoder(x, edge_index_a, head)
            x_a = self.activation(x_a).to(device)
            m1 = self.activation(m1).to(device)

            x_b, m2 = encoder(x, edge_index_b, head)
            x_b = self.activation(x_b).to(device)
            m2 = self.activation(m2).to(device)

        return x_a, x_b, m1, m2

    def forward(self, x, N, pos_edge_index, neg_edge_index, head):
        con_pos_edge_index, con_neg_edge_index, sig_pos_edge_index, sig_neg_edge_index = self.generate_view(N,
                                                                                                            pos_edge_index,
                                                                                                            neg_edge_index)

        pos_x_con, pos_x_sig, pos_m1, pos_m2 = self.encode(con_pos_edge_index, sig_pos_edge_index, x, head)
        neg_x_con, neg_x_sig, neg_m1, neg_m2 = self.encode(con_neg_edge_index, sig_neg_edge_index, x, head)

        x_concat = torch.concat((pos_x_con, pos_x_sig, neg_x_con, neg_x_sig), dim=1)
        m = torch.concat((pos_m1, pos_m2, neg_m1, neg_m2))
        return m, x_concat, pos_x_con, pos_x_sig, neg_x_con, neg_x_sig

    def similarity_score(self, x_a, x_b):
        """compute the similarity score : exp(\frac{sim_{imim'}}{\tau})"""

        sim_score = torch.bmm(x_a.view(x_a.shape[0], 1, x_a.shape[1]),
                              x_b.view(x_b.shape[0], x_b.shape[1], 1))

        return torch.exp(torch.div(sim_score, self.params.tau))

    def compute_per_loss(self, x_a, x_b):
        """inter-contrastive"""

        numerator = self.similarity_score(x_a, x_b)  # exp(\frac{sim_{imim'}}{\tau})

        denominator = torch.mm(x_a.view(x_a.shape[0], x_a.shape[1]),
                               x_b.transpose(0, 1))  # similarity value for (im, jm')

        denominator[np.arange(x_a.shape[0]), np.arange(x_a.shape[0])] = 0  # (im, im') = 0

        denominator = torch.sum(torch.exp(torch.div(denominator, self.params.tau)),
                                dim=1)  # \sum_j exp(\frac{sim_{imjm'}}{\tau})

        # -\frac{1}{I} \sum_i log(\frac{numerator}{denominator})
        return torch.mean(-torch.log(torch.div(numerator, denominator)))

    def compute_cross_loss(self, x, pos_x_a, pos_x_b, neg_x_a, neg_x_b):
        """intra-contrastive"""

        pos = self.similarity_score(x, pos_x_a) + self.similarity_score(x, pos_x_b)  # numerator

        neg = self.similarity_score(x, neg_x_a) + self.similarity_score(x, neg_x_b)  # denominator

        # -\frac{1}{I} \sum_i log(\frac{numerator}{denominator})
        return torch.mean(-torch.log(torch.div(pos, neg)))

    def compute_contrastive_loss(self, x, pos_x_con, pos_x_sig, neg_x_con, neg_x_sig):
        """contrastive-loss"""
        # x reduce dimention to feature_dim
        # self.x = self.transform(x.to(torch.float32)).to(device)
        self.x = self.transform(x).to(device)
        # Normalization
        self.x = F.normalize(self.x, p=2, dim=1)

        pos_x_con = F.normalize(pos_x_con, p=2, dim=1)
        pos_x_sig = F.normalize(pos_x_sig, p=2, dim=1)

        neg_x_con = F.normalize(neg_x_con, p=2, dim=1)
        pos_x_sig = F.normalize(pos_x_sig, p=2, dim=1)

        # inter-loss
        inter_loss_train_pos = self.compute_per_loss(pos_x_con, pos_x_sig)
        inter_loss_train_neg = self.compute_per_loss(neg_x_con, pos_x_sig)

        inter_loss = inter_loss_train_pos + inter_loss_train_neg

        # intra-loss
        intra_loss_train = self.compute_cross_loss(self.x, pos_x_con, pos_x_sig, neg_x_con, pos_x_sig)

        intra_loss = intra_loss_train

        # (1-\alpha) inter + \alpha intra
        return (1 - self.params.alpha) * inter_loss + self.params.alpha * intra_loss

    def predict(self, x_concat, src_id, dst_id):
        src_x = x_concat[src_id]
        dst_x = x_concat[dst_id]

        return self.predictor(src_x, dst_x)

    def compute_label_loss(self, score, y):
        pos_weight = torch.tensor([(y == 0).sum().item() / (y == 1).sum().item()] * y.shape[0]).to(device)
        return F.binary_cross_entropy_with_logits(score, y, pos_weight=pos_weight)

    @torch.no_grad()
    def test(self, pred_y, y):
        """test method, return acc auc f1"""
        pred = pred_y.cpu().numpy()
        test_y = y.cpu().numpy()

        # thresholds
        pred[pred >= 0] = 1
        pred[pred < 0] = 0

        acc = accuracy_score(test_y, pred)
        auc = roc_auc_score(test_y, pred)
        f1 = f1_score(test_y, pred)
        micro_f1 = f1_score(test_y, pred, average="micro")
        macro_f1 = f1_score(test_y, pred, average="macro")

        return acc, auc, f1, micro_f1, macro_f1

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

        pos_val = torch.full((pos_edge_index.size(1), ), 2, dtype=torch.float)
        neg_val = torch.full((neg_edge_index.size(1), ), 0, dtype=torch.float)
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
        svd = TruncatedSVD(n_components=self.in_channels, n_iter=128)
        svd.fit(A)
        x = svd.components_.T
        return torch.from_numpy(x).to(torch.float).to(pos_edge_index.device)