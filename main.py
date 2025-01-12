import copy
import math
import random
import time
from core.test import compute_acc, compute_local_test_accuracy
import numpy as np
import torch
import torch.optim as optim

from core.config import get_args
from core.utils import aggregation_by_graph, update_graph_matrix_neighbor, GraphConstruct, edgeindex_construct, \
    homocal, load_dataset, update_graph_matrix_neighbor_K, aggregation_by_graph_K, optimize_graph_matrix_K, \
    optimize_graph_matrix_overall_K
from core.model import simplecnn, textcnn, GFK
from core.attack import *
from sklearn.decomposition import PCA
import glog as log
import sys
import os
import json
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from datetime import datetime


def load_data(client_id, args):
    partition = [torch_load(
        args.base_path,
        f'{args.dataset}_{args.mode}/{args.n_clients}/partition_{client_id}.pt'
    )['client_data']][0]

    return partition


def accuracy(preds, targets):
    if targets.size(0) == 0: return 1.0
    with torch.no_grad():
        if args.dataset in ['Minesweeper', 'Tolokers', 'Questions']:
            acc = roc_auc_score(targets.cpu().detach().numpy(), preds.cpu().detach().numpy())
        else:
            preds = preds.max(1)[1]
            acc = preds.eq(targets).sum().item() / targets.size(0)
    return acc


def validate(net, batch, features, mode='test'):
    with torch.no_grad():
        target, pred, loss = [], [], []
        batch = batch.cuda()
        mask = batch.test_mask if mode == 'test' else batch.val_mask
        y_hat, lss = validation_step(net, features, batch, mask)
        pred.append(y_hat[mask])
        target.append(batch.y[mask])
        loss.append(lss)
        acc = accuracy(torch.stack(pred).view(-1, args.n_clss), torch.stack(target).view(-1))
    return acc, np.mean(loss)


def validation_step(model, features, batch, mask=None):
    model.eval()
    model.cuda()
    y_hat = model(features.cuda())

    if torch.sum(mask).item() == 0: return y_hat, 0.0
    if args.dataset in ['Minesweeper', 'Tolokers', 'Questions']:
        lss = F.binary_cross_entropy_with_logits(y_hat[mask].view(-1), batch.y[mask].view(-1))
    else:
        lss = F.cross_entropy(y_hat[mask], batch.y[mask])

    return y_hat, lss.item()


def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)
    return torch.load(fpath, map_location=torch.device('cpu'))


def pca_torch(data, n_components):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    data_mean = torch.mean(data_tensor, dim=0)
    data_tensor -= data_mean

    U, S, V = torch.svd(data_tensor)

    principal_components = torch.mm(data_tensor, V[:, :n_components])

    return principal_components, V[:, :n_components].T


def local_train_pfedgraph(args, local_models_optimizer, nets_this_round, cluster_models, best_val_acc_list,
                          best_test_acc_list, benign_client_list, round):
    principal_list = []

    principal_list2 = []

    nodes_on_each_clients_list = []

    local_models_optimizer_state = {}

    principal_list_K = torch.zeros([args.K, args.n_clients, k_principal, args.n_feat], dtype=torch.float32)
    principal_list2_K = torch.zeros([args.K, args.n_clients, k_principal, args.n_feat], dtype=torch.float32)

    for net_id, net in nets_this_round.items():

        optimizer = local_models_optimizer[net_id]

        batch = load_data(net_id, args)

        edge_index, feat, label = batch.edge_index, batch.x, batch.y
        train_mask, val_mask, test_mask = batch.train_mask, batch.val_mask, batch.test_mask

        num_nodes = label.shape[0]
        graph = GraphConstruct(edge_index, num_nodes)

        LP, _, _ = edgeindex_construct(edge_index, num_nodes)
        feat = torch.FloatTensor(feat)

        homoratio = homocal(graph, train_mask, label)

        features, dim, HM_l_cat, H_k_l_cat, HM_l, H_k_l = load_dataset(LP, feat, args.K, args.tau, homoratio,
                                                                       args.plain)
        features = features.view(-1, args.K + 1, dim)

        nodes_on_each_clients_list.append(num_nodes)

        net.cuda()
        net.train()
        for iteration in range(args.n_eps):
            optimizer.zero_grad()
            features = features.cuda()
            y_hat = net(features[train_mask])

            if args.dataset in ['Minesweeper', 'Tolokers', 'Questions']:
                train_lss = F.binary_cross_entropy_with_logits(y_hat.view(-1),
                                                               batch.y.cuda()[batch.train_mask].view(-1))
            else:
                train_lss = F.cross_entropy(y_hat, batch.y.cuda()[batch.train_mask])

            train_lss.backward()
            optimizer.step()

        if net_id in benign_client_list:
            val_local_acc, val_local_lss = validate(net, batch, features.cuda(), mode='valid')
            test_local_acc, test_local_lss = validate(net, batch, features.cuda(), mode='test')
            if val_local_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_local_acc
                best_test_acc_list[net_id] = test_local_acc
            log.info('>> Client %d test | Test Acc: %.4f | Test Loss: %.5f', net_id, test_local_acc, test_local_lss)
        log.info('>> concatenate')
        if net_id in benign_client_list:
            if round < 15:
                principal_components, eig_vectors = pca_torch(H_k_l_cat, k_principal)
                principal_list.append(eig_vectors.cpu().numpy())

                principal_components2, eig_vectors2 = pca_torch(HM_l_cat, k_principal)
                principal_list2.append(eig_vectors2.cpu().numpy())

                for idss in range(args.K):
                    principal_components, eig_vectors = pca_torch(H_k_l[idss], k_principal)
                    principal_list_K[idss, net_id, :, :] = eig_vectors

                    principal_components2, eig_vectors2 = pca_torch(HM_l[idss], k_principal)
                    principal_list2_K[idss, net_id, :, :] = eig_vectors2

        log.info('>> concatenate end')
        net.to('cpu')

        local_models_optimizer_state[net_id] = optimizer.state_dict()

    principal_list_K = principal_list_K.reshape(args.K, args.n_clients, -1)

    principal_list2_K = principal_list2_K.reshape(args.K, args.n_clients, -1)

    return principal_list, principal_list2, principal_list_K, principal_list2_K, np.array(best_test_acc_list)[
        np.array(benign_client_list)].mean(), nodes_on_each_clients_list, local_models_optimizer_state


args, cfg = get_args()
if args.dataset == 'Cora':
    args.n_feat = 1433
    args.n_clss = 7
elif args.dataset == 'CiteSeer':
    args.n_feat = 3703
    args.n_clss = 6
elif args.dataset == 'PubMed':
    args.n_feat = 500
    args.n_clss = 3
elif args.dataset == 'Computers':
    args.n_feat = 767
    args.n_clss = 10
elif args.dataset == 'Photo':
    args.n_feat = 745
    args.n_clss = 8
elif args.dataset == 'ogbn-arxiv':
    args.n_feat = 128
    args.n_clss = 40
elif args.dataset == 'Roman-empire':
    args.n_feat = 300
    args.n_clss = 18
elif args.dataset == 'Amazon-ratings':
    args.n_feat = 300
    args.n_clss = 5
elif args.dataset == 'Minesweeper':
    args.n_feat = 7
    args.n_clss = 1
elif args.dataset == 'Tolokers':
    args.n_feat = 10
    args.n_clss = 1
elif args.dataset == 'Questions':
    args.n_feat = 301
    args.n_clss = 1
elif args.dataset == 'Actor':
    args.n_feat = 932
    args.n_clss = 5
elif args.dataset == 'squirrel':
    args.n_feat = 2089
    args.n_clss = 5
elif args.dataset == 'chameleon':
    args.n_feat = 2325
    args.n_clss = 5
elif args.dataset == 'texas':
    args.n_feat = 1703
    args.n_clss = 5
elif args.dataset == 'cornell':
    args.n_feat = 1703
    args.n_clss = 5
elif args.dataset == 'wisconsin':
    args.n_feat = 1703
    args.n_clss = 5

seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

k_principal = int(args.k_principal)

n_party_per_round = args.n_clients
party_list = [i for i in range(args.n_clients)]
party_list_rounds = []
if n_party_per_round != args.n_clients:
    for i in range(args.comm_round):
        party_list_rounds.append(random.sample(party_list, n_party_per_round))
else:
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)

benign_client_list = random.sample(party_list, int(args.n_clients * (1 - args.attack_ratio)))
benign_client_list.sort()
log.info('>> -------- Benign --------', benign_client_list)

global_model = GFK(level=args.K,
                   nfeat=args.n_feat,
                   nlayers=args.n_layers,
                   nhidden=args.n_latentdims,
                   nclass=args.n_clss,
                   dropoutC=args.dpC,
                   dropoutM=args.dpM,
                   bias=args.bias,
                   sole=args.sole).cuda()

global_parameters = global_model.state_dict()
local_models = []
best_val_acc_list, best_test_acc_list = [], []
dw = []
for i in range(cfg['client_num']):
    local_models.append(GFK(level=args.K,
                            nfeat=args.n_feat,
                            nlayers=args.n_layers,
                            nhidden=args.n_latentdims,
                            nclass=args.n_clss,
                            dropoutC=args.dpC,
                            dropoutM=args.dpM,
                            bias=args.bias,
                            sole=args.sole).cuda())
    dw.append({key: torch.zeros_like(value) for key, value in
               local_models[i].named_parameters()})
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)

graph_matrix = torch.ones(len(local_models), len(local_models)) / (len(local_models) - 1)
graph_matrix[range(len(local_models)), range(len(local_models))] = 0

graph_matrix_K = torch.ones(args.K, len(local_models), len(local_models)) / (len(local_models) - 1)
graph_matrix_K[:, range(len(local_models)), range(len(local_models))] = 0

B = torch.nn.Parameter(torch.zeros((args.K, args.n_clients)))

B_overall = torch.nn.Parameter(torch.zeros((args.n_clients)))

one_vector = torch.ones((args.n_clients, 1)).cuda()

local_models_optimizer = {}

for i, net in enumerate(local_models):
    net.load_state_dict(global_parameters)
    net.cuda()

    if args.optimizer == 'adam':
        opt_tmp = optim.Adam([{
            'params': net.mlp.parameters(),
            'weight_decay': args.wd1,
            'lr': args.lr1
        }, {
            'params': net.comb.parameters(),
            'weight_decay': args.wd2,
            'lr': args.lr2
        }])
    elif args.optimizer == 'adamw':
        opt_tmp = optim.AdamW([{
            'params': net.mlp.parameters(),
            'weight_decay': args.wd1,
            'lr': args.lr1
        }, {
            'params': net.comb.parameters(),
            'weight_decay': args.wd2,
            'lr': args.lr2
        }])

    local_models_optimizer[i] = opt_tmp

opt_server_K = {}
for ete in range(args.K):
    optimizer_server = torch.optim.Adam([B], lr=args.lr1, weight_decay=args.wd1)
    opt_server_K[ete] = optimizer_server

optimizer_server_overall = torch.optim.Adam([B_overall], lr=args.lr1, weight_decay=args.wd1)

cluster_model_vectors = {}
total_round = cfg["comm_round"]
for round in range(total_round):
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction < 1.0:
        log.info('>> Clients in this round : %d', party_list_this_round)
    nets_this_round = {k: local_models[k] for k in party_list_this_round}
    nets_param_start = {k: copy.deepcopy(local_models[k]) for k in party_list_this_round}
    principal_list, principal_list2, principal_list_K, principal_list2_K, mean_personalized_acc, nodes_on_each_clients_list, local_models_optimizer_state = local_train_pfedgraph(
        args,
        local_models_optimizer,
        nets_this_round,
        cluster_model_vectors,
        best_val_acc_list,
        best_test_acc_list,
        benign_client_list,
        round)

    for idss in local_models_optimizer_state:
        local_models_optimizer[idss].load_state_dict(local_models_optimizer_state[idss])

        for state in local_models_optimizer[idss].state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    if k != 'step':
                        state[k] = v.cuda()

    total_data_points = sum(nodes_on_each_clients_list)
    fed_avg_freqs = {k: nodes_on_each_clients_list[i] / total_data_points for i, k in enumerate(party_list_this_round)}

    manipulate_gradient(args, None, nets_this_round, benign_client_list, nets_param_start)
    if round / total_round > args.alpha_bound:
        matrix_alpha = 0
    else:
        matrix_alpha = args.matrix_alpha

    if round < 15:
        graph_matrix_K, optimizer_server_state = optimize_graph_matrix_K(graph_matrix_K.cuda(), principal_list_K.cuda(),
                                                                         principal_list2_K.cuda(), B.cuda(), args.gamma,
                                                                         opt_server_K,
                                                                         one_vector, args.matrix_beta,
                                                                         args.matrix_alpha, args)

        graph_matrix = optimize_graph_matrix_overall_K(graph_matrix.cuda(), principal_list,
                                                       principal_list2, B_overall.cuda(), args.gamma,
                                                       optimizer_server_overall,
                                                       one_vector, args.matrix_beta, args.matrix_alpha, args)

    cluster_model_vectors = aggregation_by_graph_K(graph_matrix, graph_matrix_K, nets_this_round, global_parameters)

    log.info('>> (Current) Round %d | Local Per: %.4f', round, mean_personalized_acc)
