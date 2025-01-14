import torch
import numpy as np
import copy
import cvxpy as cp
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import glog as log
from torch_geometric.utils import to_undirected, is_undirected
import scipy.sparse as sp
import time
import gc
import math

from FedSaC_code.config import get_args


def compute_local_test_accuracy(model, dataloader, data_distribution):
    model.eval()

    toatl_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    model.cuda()
    generalized_total, generalized_correct = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = model(x)
            _, pred_label = torch.max(out.data, 1)
            correct_filter = (pred_label == target.data)
            generalized_total += x.data.size()[0]
            generalized_correct += correct_filter.sum().item()
            for i, true_label in enumerate(target.data):
                toatl_label_num[true_label] += 1
                if correct_filter[i]:
                    correct_label_num[true_label] += 1
    personalized_correct = (correct_label_num * data_distribution).sum()
    personalized_total = (toatl_label_num * data_distribution).sum()

    model.to('cpu')
    return personalized_correct / personalized_total, generalized_correct / generalized_total


def cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric):
    model_similarity_matrix = torch.zeros((len(nets_this_round), len(nets_this_round)))
    index_clientid = list(nets_this_round.keys())
    for i in range(len(nets_this_round)):
        model_i = nets_this_round[index_clientid[i]].state_dict()
        for key in dw[index_clientid[i]]:
            dw[index_clientid[i]][key] = model_i[key].cuda() - initial_global_parameters[key]
    for i in range(len(nets_this_round)):
        for j in range(i, len(nets_this_round)):
            if similarity_matric == "all":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten_all(dw[index_clientid[i]]).unsqueeze(0),
                                                               weight_flatten_all(dw[index_clientid[j]]).unsqueeze(0))
                if diff < - 0.9:
                    diff = - 1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff
            elif similarity_matric == "fc":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten(dw[index_clientid[i]]).unsqueeze(0),
                                                               weight_flatten(dw[index_clientid[j]]).unsqueeze(0))
                if diff < - 0.9:
                    diff = - 1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff

    return model_similarity_matrix


def update_graph_matrix_neighbor(graph_matrix, nets_this_round, initial_global_parameters, principal_list,
                                 principal_list2, dw, fed_avg_freqs, lambda_1,
                                 lambda_2, complementary_metric, similarity_metric):
    index_clientid = list(nets_this_round.keys())
    model_complementary_matrix = cal_complementary(nets_this_round, principal_list, complementary_metric)

    model_complementary_matrix2 = cal_complementary(nets_this_round, principal_list2, complementary_metric)

    graph_matrix = optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_complementary_matrix,
                                                    model_complementary_matrix2, lambda_1, lambda_2, fed_avg_freqs)
    return graph_matrix


def cal_complementary(nets_this_round, principal_list, complementary_metric):
    model_complementary_matrix = torch.zeros((len(nets_this_round), len(nets_this_round)), device='cuda')
    index_clientid = list(nets_this_round.keys())

    k = principal_list[0].shape[0]

    for i in range(len(nets_this_round)):
        for j in range(i, len(nets_this_round)):
            if complementary_metric == "PA":
                phi = compute_principal_angles(principal_list[i], principal_list[j])
                principal_angle = torch.cos((1 / k) * torch.sum(phi))
                model_complementary_matrix[i, j] = principal_angle
                model_complementary_matrix[j, i] = principal_angle

    return model_complementary_matrix


def update_graph_matrix_neighbor_K(graph_matrix_K, nets_this_round, principal_list_K, principal_list2_K, fed_avg_freqs,
                                   lambda_1,
                                   lambda_2, complementary_metric):
    for ids in range(graph_matrix_K.shape[0]):
        index_clientid = list(nets_this_round.keys())
        model_complementary_matrix = cal_complementary_k(nets_this_round, principal_list_K[ids], complementary_metric)

        model_complementary_matrix2 = cal_complementary_k(nets_this_round, principal_list2_K[ids], complementary_metric)

        graph_matrix_K[ids] = optimizing_graph_matrix_neighbor(graph_matrix_K[ids], index_clientid,
                                                               model_complementary_matrix, model_complementary_matrix2,
                                                               lambda_1, lambda_2, fed_avg_freqs)

    return graph_matrix_K


def cal_complementary_k(nets_this_round, principal_list, complementary_metric):
    model_complementary_matrix = torch.zeros((len(nets_this_round), len(nets_this_round)), device='cuda')
    index_clientid = list(nets_this_round.keys())

    k = principal_list.shape[1]

    for i in range(len(nets_this_round)):
        for j in range(i, len(nets_this_round)):
            if complementary_metric == "PA":
                phi = compute_principal_angles_k(principal_list[i], principal_list[j])
                principal_angle = torch.cos((1 / k) * torch.sum(phi))
                model_complementary_matrix[i, j] = principal_angle
                model_complementary_matrix[j, i] = principal_angle

    return model_complementary_matrix


def compute_principal_angles(A, B):
    assert A.shape[0] == B.shape[0], "A and B must have the same number of vectors"

    k = A.shape[0]

    A = torch.tensor(A, dtype=torch.float32).cuda()
    B = torch.tensor(B, dtype=torch.float32).cuda()

    norm_A = torch.norm(A, dim=1, keepdim=True)
    norm_B = torch.norm(B, dim=1)

    dot_product = torch.mm(A, B.T)
    cosine_matrix = dot_product / (norm_A * norm_B + 1e-10)

    cos_phi_values = []

    for _ in range(k):
        flat_idx = torch.argmax(cosine_matrix)
        i, j = divmod(flat_idx.item(), cosine_matrix.size(1))

        cos_phi_values.append(cosine_matrix[i, j].item())

        cosine_matrix[i, :] = -float('inf')
        cosine_matrix[:, j] = -float('inf')

    phi = torch.arccos(torch.clamp(torch.tensor(cos_phi_values, device='cuda'), -1, 1))

    return phi


def compute_principal_angles_k(A, B):
    assert A.shape[0] == B.shape[0], "A and B must have the same number of vectors"

    k = A.shape[0]

    norm_A = torch.norm(A, dim=1, keepdim=True)
    norm_B = torch.norm(B, dim=1)

    dot_product = torch.mm(A, B.T)
    cosine_matrix = dot_product / (norm_A * norm_B + 1e-10)

    cos_phi_values = []

    for _ in range(k):
        flat_idx = torch.argmax(cosine_matrix)
        i, j = divmod(flat_idx.item(), cosine_matrix.size(1))
        cos_phi_values.append(cosine_matrix[i, j].item())
        cosine_matrix[i, :] = -float('inf')
        cosine_matrix[:, j] = -float('inf')

    phi = torch.arccos(torch.clamp(torch.tensor(cos_phi_values, device='cuda'), -1, 1))

    return phi


def optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_complementary_matrix,
                                     model_complementary_matrix2, lambda_1, lambda_2, fed_avg_freqs):
    n = model_complementary_matrix.shape[0]
    p = np.array(list(fed_avg_freqs.values()))
    P = np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    for i in range(model_complementary_matrix.shape[0]):
        model_complementary_vector = model_complementary_matrix[i]

        model_complementary_vector2 = model_complementary_matrix2[i]

        c = model_complementary_vector.cpu().numpy()
        c2 = model_complementary_vector2.cpu().numpy()

        q = lambda_1 * c - 2 * p - lambda_2 * c2
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                          [G @ x <= h,
                           A @ x == b]
                          )
        prob.solve()

        graph_matrix[index_clientid[i], index_clientid] = torch.Tensor(x.value)
    return graph_matrix


def weight_flatten(model):
    params = []
    for k in model:
        if 'fc' in k:
            params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params


def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params


def weight_flatten_all_k(model):
    params = []
    for k in range(len(model)):
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params


def aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_w):
    tmp_client_state_dict = {}
    cluster_model_vectors = {}
    for client_id in nets_this_round.keys():
        tmp_client_state_dict[client_id] = copy.deepcopy(global_w)
        cluster_model_vectors[client_id] = torch.zeros_like(weight_flatten_all(global_w))
        for key in tmp_client_state_dict[client_id]:
            tmp_client_state_dict[client_id][key] = torch.zeros_like(
                tmp_client_state_dict[client_id][key])

    for client_id in nets_this_round.keys():
        tmp_client_state = tmp_client_state_dict[client_id]
        cluster_model_state = cluster_model_vectors[client_id]
        aggregation_weight_vector = graph_matrix[client_id]

        for neighbor_id in nets_this_round.keys():
            net_para = nets_this_round[neighbor_id].state_dict()
            for key in tmp_client_state:
                tmp_client_state[key] += net_para[key].cuda() * aggregation_weight_vector[neighbor_id]

        for neighbor_id in nets_this_round.keys():
            net_para = weight_flatten_all(nets_this_round[neighbor_id].state_dict())
            cluster_model_state += net_para.cuda() * (
                    aggregation_weight_vector[neighbor_id] / torch.linalg.norm(net_para))

    for client_id in nets_this_round.keys():
        nets_this_round[client_id].load_state_dict(tmp_client_state_dict[client_id])

    return cluster_model_vectors


def aggregation_by_graph_K(graph_matrix, graph_matrix_K, nets_this_round, global_w):
    tmp_client_state_dict = {}
    cluster_model_vectors = {}
    for client_id in nets_this_round.keys():
        tmp_client_state_dict[client_id] = copy.deepcopy(global_w)
        cluster_model_vectors[client_id] = torch.zeros_like(weight_flatten_all(global_w))
        for key in tmp_client_state_dict[client_id]:
            tmp_client_state_dict[client_id][key] = torch.zeros_like(
                tmp_client_state_dict[client_id][key])

    for client_id in nets_this_round.keys():
        tmp_client_state = tmp_client_state_dict[client_id]
        cluster_model_state = cluster_model_vectors[client_id]
        aggregation_weight_vector = graph_matrix[client_id]

        for neighbor_id in nets_this_round.keys():
            net_para = nets_this_round[neighbor_id].state_dict()
            for key in tmp_client_state:
                if key != 'comb.comb_weight':
                    tmp_client_state[key] += net_para[key].cuda() * aggregation_weight_vector[neighbor_id]
                else:
                    tmp_client_state[key][:, 0, :] += net_para[key][:, 0, :].cuda() * aggregation_weight_vector[
                        neighbor_id]

                    for idsss in range(net_para[key].shape[1] - 1):
                        tmp_client_state[key][:, idsss + 1, :] += net_para[key][:, idsss + 1, :].cuda() * \
                                                                  graph_matrix_K[idsss, client_id, neighbor_id]

        for neighbor_id in nets_this_round.keys():
            model_paras = []
            net_para = nets_this_round[neighbor_id].state_dict()
            for key in tmp_client_state:
                if key != 'comb.comb_weight':
                    model_paras.append(net_para[key])

            net_para1 = weight_flatten_all_k(model_paras)

            net_para = nets_this_round[neighbor_id].state_dict()
            model_paras_tensors_l = []
            for key in tmp_client_state:
                if key == 'comb.comb_weight':
                    net_paras = weight_flatten_all_k(net_para[key][:, 0, :])
                    model_paras_tensors_l.append(net_paras)
                    for idx in range(net_para[key].shape[1] - 1):
                        net_paras = weight_flatten_all_k(net_para[key][:, idx + 1, :])
                        model_paras_tensors_l.append(net_paras)

            model_paras_tensors = torch.cat(model_paras_tensors_l)
            net_para = torch.cat((model_paras_tensors, net_para1), dim=0)
            cluster_model_state += net_para.cuda() * (
                    aggregation_weight_vector[neighbor_id] / torch.linalg.norm(net_para))

    for client_id in nets_this_round.keys():
        nets_this_round[client_id].load_state_dict(tmp_client_state_dict[client_id])

    return cluster_model_vectors


def compute_acc(net, test_data_loader):
    net.eval()
    correct, total = 0, 0
    net.cuda()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            _, pred_label = torch.max(out.data, 1)
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()
    net.to('cpu')
    return correct / float(total)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def compute_loss(net, test_data_loader):
    net.eval()
    loss, total = 0, 0
    net.cuda()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            loss += torch.nn.functional.cross_entropy(out, target).item()
            total += x.data.size()[0]
    net.to('cpu')
    return loss / float(total)


def GraphConstruct(edge_index, n):
    graph = []
    for i in range(n):
        edge = []
        graph.append(edge)
    m = edge_index.shape[1]
    for i in range(m):
        u, v = edge_index[0][i], edge_index[1][i]
        graph[u].append(v)
    return graph


def PropMatrix(adj):
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    del row_sum
    gc.collect()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    t = time.time()
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return adj


def edgeindex_construct(edge_index, num_nodes):
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)
    edge_index = edge_index.numpy()
    num_edges = edge_index[0].shape[0]
    data = np.array([1] * num_edges)
    adj = sp.coo_matrix((data, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)).tocsr()
    t = time.time()
    adj = PropMatrix(adj)
    Propagate_matrix_time = time.time() - t
    t = time.time()
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    sparse_mx_time = time.time() - t

    return adj, Propagate_matrix_time, sparse_mx_time


def homocal(graph, train_msk, labels):
    n = labels.shape[0]
    edge = 0.0
    cnt = 0.0
    for node in range(train_msk.sum().item()):
        for nei in graph[node]:
            if train_msk[nei]:
                edge += 1.0
                if labels[node] == labels[nei]:
                    cnt += 1.0
    if edge == 0:
        return 0.75
    return cnt / edge


def load_dataset(LP, feat, K=6, tau=1.0, homo_ratio=0.6, plain=False):
    num_nodes, dim = feat.shape

    cosval = math.cos(math.pi * (1.0 - homo_ratio) / 2.0)
    if not plain:
        t1 = time.time()
        norm = torch.norm(feat, dim=0)
        norm = torch.clamp(norm, 1e-8)
        last = feat / norm
        second = torch.zeros_like(last)
        basis_sum = torch.zeros_like(last)
        HM = torch.zeros_like(last)
        HM += feat
        basis_sum += last
        features = [feat]
        HM_l = []
        H_k_l = []
        for k in range(1, K + 1):
            V_k = torch.spmm(LP, last)
            HM = torch.spmm(LP, HM)
            project_1 = torch.einsum('nd,nd->d', V_k, last)
            project_2 = torch.einsum('nd,nd->d', V_k, second)
            V_k -= (project_1 * last + project_2 * second)
            norm = torch.norm(V_k, dim=0)
            norm = torch.clamp(norm, 1e-8)
            V_k /= norm
            H_k = basis_sum / k
            Tf = torch.sqrt(
                torch.square(torch.einsum('nd,nd->d', H_k, features[-1]) / cosval) - ((k - 1) * cosval + 1) / k)
            torch.nan_to_num_(Tf, nan=0.0)
            H_k += torch.mul(Tf, V_k)
            norm = torch.norm(H_k, dim=0)
            norm = torch.clamp(norm, 1e-8)
            H_k /= norm
            norm = torch.norm(HM, dim=0)
            norm = torch.clamp(norm, 1e-8)
            features.append(HM * tau + H_k * (1.0 - tau))
            HM_l.append(HM)
            H_k_l.append(H_k)
            basis_sum += H_k
            second = last
            last = V_k
        features_time = time.time() - t1
        del last, second, LP
        gc.collect()
        features = torch.cat(features, 1)
        HM_l_cat = torch.cat(HM_l, 1)
        H_k_l_cat = torch.cat(H_k_l, 1)
    else:
        t1 = time.time()
        features = [feat]
        basis = feat
        for i in range(1, K + 1):
            basis = torch.spmm(LP, basis)
            features.append(basis)
        features_time = time.time() - t1
        del basis, LP
        gc.collect()
        features = torch.cat(features, 1)
    return features, dim, HM_l_cat, H_k_l_cat, HM_l, H_k_l


def Update_R(W, P):
    W_sym = (W + W.T) / 2

    degree_vector = torch.sum(W_sym, axis=1)
    D = torch.diag(degree_vector)

    L = D - W_sym

    M = torch.mm(torch.mm(P.T, L), P)

    m_i = torch.diag(M) + 1e-10
    m_i_inverse = 1 / m_i
    m_i_sum = torch.sum(m_i_inverse)

    scaling_factor = 10000
    r = (m_i_inverse / m_i_sum) * scaling_factor

    return r, L


def Update_S(Q, L):
    N = torch.mm(torch.mm(Q.T, L), Q)
    n_i = torch.diag(N) + 1e-10
    n_i_inverse = 1 / n_i
    n_i_sum = torch.sum(n_i_inverse)
    scaling_factor = 10000
    s = (n_i_inverse / n_i_sum) * scaling_factor
    return s


def Update_W(optimizer_s, W, r, s, P, Q, B, gamma, one_vector, similarity_para, complementarity_para, args):
    M = P.shape[0]
    R = torch.diag(r)
    S = torch.diag(s)
    P_R = torch.mm(R, P.T)
    Q_S = torch.mm(S, Q.T)
    if P.shape[1] == (args.k_principal * args.K * args.n_feat):
        P_R = P_R.reshape(-1, args.k_principal, args.n_feat * args.K)
        Q_S = Q_S.reshape(-1, args.k_principal, args.n_feat * args.K)
    else:
        P_R = P_R.reshape(-1, args.k_principal, args.n_feat)
        Q_S = Q_S.reshape(-1, args.k_principal, args.n_feat)
    t_ij = torch.zeros((M, M)).cuda()
    for i in range(M):
        for j in range(i, M):
            phi = compute_principal_angles_k(P_R[i], P_R[j])
            principal_angle = torch.cos((1 / 3) * torch.sum(phi))
            phi_yizhi = compute_principal_angles_k(Q_S[i], Q_S[j])
            principal_angle_yizhi = torch.cos((1 / 3) * torch.sum(phi_yizhi))
            t_ij[i, j] = complementarity_para * principal_angle_yizhi - similarity_para * principal_angle
            t_ij[j, i] = complementarity_para * principal_angle_yizhi - similarity_para * principal_angle
    h_ij = torch.zeros((M, M)).cuda()
    for i in range(M):
        for j in range(i, M):
            h_ij[i, j] = 1 / M - t_ij[i, j] / (2 * gamma) + torch.mm(one_vector.T, t_ij[i, :].unsqueeze(dim=1)) / (
                    2 * M * gamma)
            h_ij[j, i] = h_ij[i, j]
    for idx in range(M):
        b_i = B[idx].unsqueeze(dim=0)
        loss = (1 / M) * torch.sum(torch.nn.functional.relu(b_i - h_ij[idx, :])) - b_i
        grad_loss = torch.autograd.grad(loss, b_i, create_graph=True)[0]
        delta_b = loss / grad_loss
        B[idx] = B[idx] - delta_b.item()
        W[idx, :] = torch.nn.functional.relu(h_ij[idx, :] - B[idx])

    return W


def optimize_graph_matrix_K(graph_matrix_K, principal_list_K, principal_list2_K, B, gamma, opt_server_K,
                            one_vector, similarity_para, complementarity_para, args):
    optimizer_server_state = {}
    for ids in range(graph_matrix_K.shape[0]):
        r, L = Update_R(graph_matrix_K[ids], principal_list2_K[ids])
        s = Update_S(principal_list_K[ids], L)
        graph_matrix_K[ids] = Update_W(opt_server_K[ids], graph_matrix_K[ids], r, s, principal_list2_K[ids],
                                       principal_list_K[ids], B[ids], gamma, one_vector, similarity_para,
                                       complementarity_para, args)
    return graph_matrix_K, optimizer_server_state


def optimize_graph_matrix_overall_K(graph_matrix, principal_list, principal_list2, B, gamma, opt,
                                    one_vector, similarity_para, complementarity_para, args):
    principal_list = torch.tensor(principal_list)
    principal_list = principal_list.reshape(graph_matrix.shape[0], -1).cuda()
    principal_list2 = torch.tensor(principal_list2)
    principal_list2 = principal_list2.reshape(graph_matrix.shape[0], -1).cuda()
    r, L = Update_R(graph_matrix, principal_list2)
    s = Update_S(principal_list, L)
    graph_matrix = Update_W(opt, graph_matrix, r, s, principal_list2,
                            principal_list, B, gamma, one_vector, similarity_para, complementarity_para, args)

    return graph_matrix
