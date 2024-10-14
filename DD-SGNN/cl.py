import os,argparse
import torch
from torch import Tensor
from torch import nn
import torch.optim as optim
import datetime, time

from utils import *
from modules import *
from layers import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='Bitcoin', help='dataset')
parser.add_argument("--hidden", type=int, default=32, help='hidden layer dimension')
parser.add_argument("--seed",type=int,default=38,help="Random seed for sklearn pre-training.")
parser.add_argument("--g_sigma", type=float, default=1, help='G deviation')
parser.add_argument("--epochs", type=int, default=3000, help='Epochs')
parser.add_argument("--patience", type=int, default=300, help='Patience')
parser.add_argument("--ablation", type=int, default=0, help='ablation mode')
parser.add_argument("--eta", type=float, default=0.1, help='adversarial constraint')
parser.add_argument("--mu", type=float, default=0.01, help='missing info constraint')
parser.add_argument("--lamda", type=float, default=0.0001, help='l2 parameter')
parser.add_argument("--k", type=int, default=30, help='num of node neighbor')
parser.add_argument("--lr", type=float, default=0.01, help='learning rate')
parser.add_argument("--id", type=int, default=0, help='gpu ids')
parser.add_argument("--a",type=float, default=1, help='head')
parser.add_argument("--b",type=float, default=1, help='tail')
parser.add_argument("--c", type=int, default=1, help='m weight')
parser.add_argument("--alpha", type=float, default=0.2, help='sgcl args')
parser.add_argument("--beta", type=float, default=0.0001, help='sgcl args')
parser.add_argument("--tau", type=float, default=0.05, help='sgcl args')
parser.add_argument("--aug_ratio", type=float, default=0.1, help='sgcl args')
parser.add_argument("--x", type=Tensor, )
args = parser.parse_args()
dataset = args.dataset

cuda = torch.cuda.is_available()
criterion = nn.BCELoss()

torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.id)

device = 'cuda' if cuda else 'cpu'
dataset = args.dataset

def normalize_output(out_feat, idx):
    sum_m = 0
    for m in out_feat.transpose(0, 1):
        sum_m += torch.mean(torch.norm(m[idx], dim=0))
    return sum_m

def train_disc(batch):
    disc.train()
    optimizer_D.zero_grad()
    _, z_h, *_ = model(features, n, train_pos_edge_index, train_neg_edge_index, True)
    _, z_t, *_ = model(features, n, train_tail_pos_edge_index, train_tail_neg_edge_index, False)

    prob_h = disc(z_h)
    prob_t = disc(z_t)

    errorD = criterion(prob_h[batch], h_labels)
    errorG = criterion(prob_t[batch], t_labels)

    L_d = (errorD + errorG)/2
    L_d.backward()
    optimizer_D.step()
    return L_d

def train(batch):
    model.train()
    optimizer.zero_grad()
    x = model.dimension_reduction()
    support_h, z_h, *other_x_h = model(x, n, train_pos_edge_index, train_neg_edge_index, True)
    support_t, z_t, *other_x_t = model(x, n, train_tail_pos_edge_index, train_tail_neg_edge_index, False)

    contrastive_loss_h = model.compute_contrastive_loss(z_h, *other_x_h)
    contrastive_loss_t = model.compute_contrastive_loss(z_t, *other_x_t)
    contrastive_loss = args.a * contrastive_loss_h + args.b * contrastive_loss_t

    src_id_h = torch.concat((train_pos_edge_index[0], train_neg_edge_index[0])).to(device)
    dst_id_h = torch.concat((train_pos_edge_index[1], train_neg_edge_index[1])).to(device)

    src_id_t = torch.concat((train_tail_pos_edge_index[0], train_tail_neg_edge_index[0])).to(device)
    dst_id_t = torch.concat((train_tail_pos_edge_index[1], train_tail_neg_edge_index[1])).to(device)

    y_train_h = torch.concat((torch.ones(train_pos_edge_index.shape[1]), torch.zeros(train_neg_edge_index.shape[1]))).to(
        device)
    y_train_t = torch.concat((torch.ones(train_tail_pos_edge_index.shape[1]), torch.zeros(train_tail_neg_edge_index.shape[1]))).to(
        device)
    score_h = model.predict(model.x, src_id_h, dst_id_h)
    score_t = model.predict(model.x, src_id_t, dst_id_t)

    label_loss_h = model.compute_label_loss(score_h, y_train_h)
    label_loss_t = model.compute_label_loss(score_t, y_train_t)
    label_loss = args.a * label_loss_h + args.b * label_loss_t

    loss = args.beta * contrastive_loss + label_loss

    m_h = normalize_output(support_h, batch)
    prob_t = disc(z_t)

    errorG = criterion(prob_t[batch], t_labels)
    L_d = errorG
    L_all = loss + args.mu * m_h - (args.eta * L_d)
    L_all.backward()
    optimizer.step()
    return L_all

def test():
    model.eval()
    with torch.no_grad():
        _, z_test, *other_x_test = model(features, n, train_pos_edge_index, train_neg_edge_index, True)
        test_src_id = torch.concat((test_pos_edge_index[0], test_neg_edge_index[0])).to(device)
        test_dst_id = torch.concat((test_pos_edge_index[1], test_neg_edge_index[1])).to(device)

        y_test = torch.concat((torch.ones(test_pos_edge_index.shape[1]), torch.zeros(test_neg_edge_index.shape[1]))).to(
            device)

        score_test = model.predict(model.x, test_src_id, test_dst_id).to(device)

        test_acc, test_auc, test_f1, micro_f1, macro_f1 = model.test(score_test, y_test)
    log2 = "Test set results: \n" + \
            "test_acc={:.4f} ".format(test_acc) + \
            "test_auc={:.4f} ".format(test_auc) + \
            "test_f1={:.4f} ".format(test_auc) + \
            "micro_f1={:.4f} ".format(micro_f1) + \
            "macro_f1={:.4f} ".format(macro_f1)
    print(log2)
    return

base_path = os.path.dirname(os.path.abspath(__file__))


data_path = os.path.join(base_path, 'data', 'Bitcoin', 'Bitcoinotc.txt')

n, train_adj, train_edge_index, train_pos_edge_index, train_neg_edge_index,train_tail_pos_edge_index, train_tail_neg_edge_index,test_pos_edge_index, test_neg_edge_index, idx = data_process.process_Dataset(args, data_path, args.k)

_, edge_index, _, _ = process_data(data_path)
degrees = torch.zeros(n, dtype=torch.long)
for i in range(edge_index.size(1)):
    src, dst = edge_index[:, i]
    degrees[src] += 1
    degrees[dst] += 1
max_degree = degrees.max().item()

idx_head = torch.LongTensor(idx[0])
idx_head = torch.LongTensor(idx_head-1)
idx_tail = torch.LongTensor(idx[1])
idx_tail = torch.LongTensor(idx_tail-1)

model = MySGCL(in_channels=32, out_channels=32, layer_num=2, params=args)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
features = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index)
args.x = features
disc = Discriminator(128).to(device)
optimizer_D = optim.Adam(disc.parameters(), lr=0.001, weight_decay=args.lamda)

if cuda:
    model = model.cuda()
    disc = disc.cuda()

    features = features.cuda()
    train_pos_edge_index = train_pos_edge_index.cuda()
    train_neg_edge_index = train_neg_edge_index.cuda()
    train_tail_pos_edge_index = train_tail_pos_edge_index.cuda()
    train_tail_neg_edge_index = train_tail_neg_edge_index.cuda()

    test_pos_edge_index = test_pos_edge_index.cuda()
    test_neg_edge_index = test_neg_edge_index.cuda()

h_labels = torch.full((len(idx_head), 1), 1.0, device=device)
t_labels = torch.full((len(idx_head), 1), 0.0, device=device)

# Train model
t_total = time.time()
auc_list = []
f1_list = []
for epoch in range(args.epochs):
    t = time.time()
    L_d = train_disc(idx_head)
    Loss_train = train(idx_head)
    test()
    log1 = 'Epoch: {:d} \n'.format(epoch + 1) + \
           'loss_train: {:.4f}  '.format(Loss_train) + \
           'L_d: {:.4f} '.format(L_d)
    print(log1)





