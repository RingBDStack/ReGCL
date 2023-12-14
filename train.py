import argparse
import os.path as osp
import time
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
from model import Encoder, Model, drop_feature
from eval import evaluation
import numpy as np


def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()

    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]

    device = edge_index_1.device

    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)

    pre_z1 = model(x_1, edge_index_1, 1, None, None)
    pre_z2 = model(x_2, edge_index_2, 2, None, None)

    z1 = model(x_1, edge_index_1, 1, pre_z1, pre_z2)
    z2 = model(x_2, edge_index_2, 2, pre_z1, pre_z2)

    loss = model.loss(z1, z2, edge_index_1, edge_index_2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model, x, edge_index, y, name, device, data, learning_rate2, weight_decay2, final=False):
    model.eval()
    z = model(x, edge_index, 1, None, None)
    return evaluation(z, y, name, device, data, learning_rate2, weight_decay2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--mode', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr2', type=float, default=1e-2)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--dfr1', type=float, default=0.1)
    parser.add_argument('--dfr2', type=float, default=0.1)
    parser.add_argument('--der1', type=float, default=0.1)
    parser.add_argument('--der2', type=float, default=0.1)
    parser.add_argument('--lv', type=int, default=1)
    parser.add_argument('--cutway', type=int, default=2)
    parser.add_argument('--cutrate', type=float, default=1.0)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--wd2', type=float, default=1e-4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_hidden', type=int, default=512)
    parser.add_argument('--num_proj_hidden', type=int, default=512)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--num_epochs', type=int, default=500)
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    learning_rate = args.lr
    learning_rate2 = args.lr
    drop_edge_rate_1 = args.der1
    drop_edge_rate_2 = args.der2
    drop_feature_rate_1 = args.dfr1
    drop_feature_rate_2 = args.dfr2
    tau = args.tau
    mode = args.mode
    nei_lv = args.lv
    cutway = args.cutway
    cutrate = args.cutrate
    num_hidden = args.num_hidden
    num_proj_hidden = args.num_proj_hidden
    activation = F.relu
    base_model = GCNConv
    num_layers = args.num_layers
    num_epochs = args.num_epochs
    weight_decay = args.wd
    weight_decay2 = args.wd2


    def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'computers', 'photo', 'CS', 'Physics']
        name = 'dblp' if name == 'DBLP' else name

        if name == 'CS' or name == 'Physics':
            return Coauthor(
                path,
                name, transform=T.NormalizeFeatures())

        if name == 'computers' or name == 'photo':
            return Amazon(
                path,
                name, transform=T.NormalizeFeatures())

        return (CitationFull if name == 'dblp' else Planetoid)(
            path,
            name, transform=T.NormalizeFeatures())

    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    # encoder
    encoder = Encoder(dataset.num_features, num_hidden, activation, mode,
                      base_model=base_model, k=num_layers, cutway=cutway, cutrate=cutrate, tau=tau).to(device)
    # model
    model = Model(encoder, num_hidden, num_proj_hidden, mode, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # add high_lv neighbors
    coo1 = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.x.shape[0])
    coo1 = coo1.toarray()
    tmp = coo1.copy()
    for i in range(2, nei_lv + 1):
        coo1 += tmp ** i
    coo1 = sp.coo_matrix(coo1)
    indices = np.vstack((coo1.row, coo1.col))
    edge_index = torch.LongTensor(indices).to(device)

    # save_path
    cur_path = osp.abspath(__file__)
    cur_dir = osp.dirname(cur_path)
    model_save_path = osp.join(cur_dir, args.dataset + '.pkl')

    start = t()
    prev = start
    if not args.test:
        for epoch in range(1, num_epochs + 1):
            loss = train(model, data.x, edge_index)
            now = t()
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                  f'this epoch {now - prev:.4f}, total {now - start:.4f}')
            prev = now
            if epoch % 10 == 0:
                print("=== Test ===")
                eval_acc = test(model, data.x, data.edge_index, data.y, args.dataset, device, data, learning_rate2,
                                weight_decay2, final=True)


    if not args.test:
        print('save_path:',model_save_path)
        torch.save(model.state_dict(), model_save_path)
        print('save_success')

    else:
        if osp.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path))
            model.eval()
        else:
            print('model not exit')
            sys.exit(0)

    print("=== Eval ===")
    accs = []
    for i in range(10):
        acc = test(model, data.x, data.edge_index, data.y, args.dataset, device, data, learning_rate2, weight_decay2,
                   final=True)
        accs.append(acc)

    accs = torch.tensor(accs)
    fin_acc=torch.mean(accs)
    fin_std=torch.std(accs)
    print('fin_accuracy',fin_acc,'fin_std',fin_std)
