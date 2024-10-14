import os,argparse
import torch
from torch import nn
import datetime, time
from torch import Tensor
from utils import *
from modules import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='Bitcoin', help='dataset')
parser.add_argument("--hidden", type=int, default=32, help='hidden layer dimension')
parser.add_argument("--seed",type=int,default=38,help="Random seed for sklearn pre-training.")
parser.add_argument("--g_sigma", type=float, default=1, help='G deviation')
parser.add_argument("--epochs", type=int, default=2000, help='Epochs')
parser.add_argument("--patience", type=int, default=300, help='Patience')
parser.add_argument("--ablation", type=int, default=0, help='ablation mode')
parser.add_argument("--eta", type=float, default=0.1, help='adversarial constraint')
parser.add_argument("--mu", type=float, default=0.001, help='missing info constraint')
parser.add_argument("--lamda", type=float, default=0.0001, help='l2 parameter')
parser.add_argument("--k", type=int, default=30, help='num of node neighbor')
parser.add_argument("--lr", type=float, default=0.001, help='learning rate')
parser.add_argument("--id", type=int, default=0, help='gpu ids')
parser.add_argument("--c", type=int, default=0.5, help='m weight')
parser.add_argument("--alpha", type=float, default=0.2, help='sgcl args')
parser.add_argument("--beta", type=float, default=0.0001, help='sgcl args')
parser.add_argument("--tau", type=float, default=0.05, help='sgcl args')
parser.add_argument("--aug_ratio", type=float, default=0.1, help='sgcl args')
parser.add_argument("--x", type=Tensor)
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

def train_cl():
    model.train()
    optimizer.zero_grad()
    x_concat, *other_x = model(features, n, train_pos_edge_index, train_neg_edge_index)
    # loss
    contrastive_loss = model.compute_contrastive_loss(x_concat, *other_x)
    # train predict
    src_id = torch.concat((train_pos_edge_index[0], train_neg_edge_index[0])).to(device)
    dst_id = torch.concat((train_pos_edge_index[1], train_neg_edge_index[1])).to(device)

    y_train = torch.concat((torch.ones(train_pos_edge_index.shape[1]), torch.zeros(train_neg_edge_index.shape[1]))).to(
        device)
    score = model.predict(model.x, src_id, dst_id)
    label_loss = model.compute_label_loss(score, y_train)
    loss = args.beta * contrastive_loss + label_loss
    loss.backward()
    optimizer.step()
    return loss

def test_cl():
    model.eval()
    with torch.no_grad():
        x_concat, *other_x = model(features, n, train_pos_edge_index, train_neg_edge_index)

        # test predict
        test_src_id = torch.concat((test_pos_edge_index[0], test_neg_edge_index[0])).to(device)
        test_dst_id = torch.concat((test_pos_edge_index[1], test_neg_edge_index[1])).to(device)

        y_test = torch.concat((torch.ones(test_pos_edge_index.shape[1]), torch.zeros(test_neg_edge_index.shape[1]))).to(
            device)
        score_test = model.predict(model.x, test_src_id, test_dst_id).to(device)
        test_acc, test_auc, test_f1, micro_f1, macro_f1 = model.test(score_test, y_test)
    log2 = "Test set results: \n" + \
           "test_acc={:.4f} ".format(test_acc) + \
           "test_auc={:.4f} ".format(test_auc) + \
           "test_f1={:.4f} ".format(test_f1) + \
           "micro_f1={:.4f} ".format(micro_f1) + \
           "macro_f1={:.4f} ".format(macro_f1)
    print(log2)
    return

def train():
    model.train()
    optimizer.zero_grad()
    z = model(features, train_pos_edge_index, train_neg_edge_index)
    loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        z_test = model(features, train_pos_edge_index, train_neg_edge_index)
        test_auc, test_f1 = model.test(z_test, test_pos_edge_index, test_neg_edge_index)
        multi_acc, single_acc = model.compute_tri_edge_accuracy(degrees, test_pos_edge_index, test_neg_edge_index, z_test)
        dsp = abs(multi_acc - single_acc)

    log2 = "Test set results: \n" + \
            "test_auc={:.4f} ".format(test_auc) + \
            "test_f1={:.4f} ".format(test_f1) + \
            "multi_acc={:.4f} ".format(multi_acc) + \
            "single_acc={:.4f} ".format(single_acc) + \
            "dsp={:.4f} ".format(dsp)
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

max_d = torch.max(degrees).item()
print("max_d is")
print(max_d)

#model = SignedGCN(in_channels=64, hidden_channels=64, num_layers=2, lamb=5).to(device)
#model = SNEA(in_dims=64, out_dims=64, layers=2, params=args).to(device)
model = SGCL(in_channels=32, out_channels=32, layer_num=2, params=args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
features = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index)
args.x = features

if cuda:
    model = model.cuda()

    features = features.cuda()
    train_pos_edge_index = train_pos_edge_index.cuda()
    train_neg_edge_index = train_neg_edge_index.cuda()
    train_tail_pos_edge_index = train_tail_pos_edge_index.cuda()
    train_tail_neg_edge_index = train_tail_neg_edge_index.cuda()

    test_pos_edge_index = test_pos_edge_index.cuda()
    test_neg_edge_index = test_neg_edge_index.cuda()

# Train model
t_total = time.time()
auc_list = []
f1_list = []
for epoch in range(args.epochs):
    t = time.time()
    Loss_train = train_cl()
    test_cl()
    log1 = 'Epoch: {:d} \n'.format(epoch + 1) + \
           'loss_train: {:.4f}  '.format(Loss_train)
    print(log1)