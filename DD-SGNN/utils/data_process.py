# Data
import scipy.sparse as sp
import torch
import numpy as np
from torch_geometric.data import Data

# split head vs tail nodes
def split_nodes(adj, k=30):
    num_links = np.sum(adj, axis=1)
    idx_train = np.where(num_links > k)[0]
    idx_valtest = np.where(num_links <= k)[0]
    np.random.shuffle(idx_valtest)
    return idx_train, idx_valtest

def link_dropout(adj, idx, k=30):
    tail_adj = adj.copy()
    num_links = np.random.randint(k, size=idx.shape[0])
    num_links += 1

    for i in range(idx.shape[0]):
        if idx[i] < tail_adj.shape[0]:
            index = tail_adj[idx[i]].nonzero()[1]
        else:
            print(f"Warning: Row index {idx[i]} is out of range for matrix with shape {tail_adj.shape}.")
        if index.size > 0:
            new_idx = np.random.choice(index, num_links[i], replace=True)
            tail_adj[idx[i]] = 0.0
            for j in new_idx:
                tail_adj[idx[i], j] = 1.0

    return tail_adj

def adj_to_index(adj):
    if not isinstance(adj, sp.coo_matrix):
        adj = adj.tocoo()
    row_indices, col_indices = adj.nonzero()
    edge_index = np.vstack((row_indices, col_indices)).T
    return edge_index

def split_edges(edge_index, test_ratio: float = 0.2):
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        mask[torch.randperm(mask.size(0))[:int(test_ratio * mask.size(0))]] = 0

        train_edge_index = edge_index[:, mask]
        test_edge_index = edge_index[:, ~mask]

        return train_edge_index, test_edge_index

def process_data(file_path):
    edge_index = []
    edge_attr = []

    with open(file_path, 'r') as f:
        for line in f:
            src, dst, attr = map(int, line.strip().split())
            edge_index.append([src, dst])
            edge_attr.append(attr)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    num_nodes = edge_index.max().item() + 1

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    pos_edge_indices, neg_edge_indices = [], []
    pos_edge_indices.append(data.edge_index[:, data.edge_attr > 0])
    neg_edge_indices.append(data.edge_index[:, data.edge_attr < 0])

    return num_nodes, edge_index, pos_edge_indices, neg_edge_indices

def process_data1(file_path):
    edge_index = []
    edge_attr = []

    with open(file_path, 'r') as f:
        for line in f:
            src, dst, attr, _ = map(int, line.strip().split(','))  # 假设每行是两个节点的编号和边属性
            edge_index.append([src, dst])
            edge_attr.append(attr)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # 假设边属性是浮点数

    num_nodes = edge_index.max().item() + 1

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    pos_edge_indices, neg_edge_indices = [], []
    pos_edge_indices.append(data.edge_index[:, data.edge_attr > 0])
    neg_edge_indices.append(data.edge_index[:, data.edge_attr < 0])

    return num_nodes, edge_index, pos_edge_indices, neg_edge_indices

def index_to_adj(edge_index, N):
    adj = sp.coo_matrix((np.ones(edge_index.shape[0]), (edge_index[:, 0], edge_index[:, 1])),
                        shape=(N+1, N+1), dtype=np.float32)
    adj = adj.tolil()
    ind = np.where(adj.todense() > 1.0)
    for i in range(ind[0].shape[0]):
        adj[ind[0][i], ind[1][i]] = 1.
    # build symmetric adjacency matrix
    adj = adj + adj.T - adj.multiply(adj.T > adj)
    adj = adj.tolil()

    for i in range(adj.shape[0]):
        adj[i, i] = 0.
    return adj

def h_t(adj, idx):
    tail_adj = link_dropout(adj, idx)
    tail_edge_index = adj_to_index(tail_adj)
    tail_edge_index = torch.tensor(tail_edge_index, dtype=torch.int64).T.contiguous()
    return tail_edge_index

def get_adj(edge_index, n):
    edge_index = torch.LongTensor(edge_index).T.contiguous()
    adj = index_to_adj(edge_index, n)
    return adj

def process_Dataset(args, file_path,k=30):
    n, edge_index, pos_edge_index, neg_edge_index = process_data(file_path)
    pos_edge_index = torch.cat(pos_edge_index, dim=1)
    neg_edge_index = torch.cat(neg_edge_index, dim=1)

    train_edge_index, test_edge_index = split_edges(edge_index)
    train_pos_edge_index, test_pos_edge_index = split_edges(pos_edge_index)
    train_neg_edge_index, test_neg_edge_index = split_edges(neg_edge_index)

    train_adj = get_adj(train_edge_index, n)
    train_pos_adj = get_adj(train_pos_edge_index, n)
    train_neg_adj = get_adj(train_neg_edge_index, n)

    idx_head, idx_tail = split_nodes(train_adj, k=k)
    train_tail_pos_edge_index = h_t(train_pos_adj, idx_head)
    train_tail_neg_edge_index = h_t(train_neg_adj, idx_head)
    return (n, train_adj,
            train_edge_index, train_pos_edge_index, train_neg_edge_index,
            train_tail_pos_edge_index, train_tail_neg_edge_index,
            test_pos_edge_index, test_neg_edge_index,
            (idx_head, idx_tail))

def process_Dataset1(args, file_path,k=30):
    n, edge_index, pos_edge_index, neg_edge_index = process_data1(file_path)
    pos_edge_index = torch.cat(pos_edge_index, dim=1)
    neg_edge_index = torch.cat(neg_edge_index, dim=1)

    train_edge_index, test_edge_index = split_edges(edge_index)
    train_pos_edge_index, test_pos_edge_index = split_edges(pos_edge_index)
    train_neg_edge_index, test_neg_edge_index = split_edges(neg_edge_index)

    train_adj = get_adj(train_edge_index, n)
    train_pos_adj = get_adj(train_pos_edge_index, n)
    train_neg_adj = get_adj(train_neg_edge_index, n)

    idx_head, idx_tail = split_nodes(train_adj, k=k)
    train_tail_pos_edge_index = h_t(train_pos_adj, idx_head)
    train_tail_neg_edge_index = h_t(train_neg_adj, idx_head)
    return (n, train_adj,
            train_edge_index, train_pos_edge_index, train_neg_edge_index,
            train_tail_pos_edge_index, train_tail_neg_edge_index,
            test_pos_edge_index, test_neg_edge_index,
            (idx_head, idx_tail))
