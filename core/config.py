import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--exp_dir', type=str, default='experiments',
                        help='Locations to save different experimental runs.')
    parser.add_argument('--dataset', type=str, default='Cora', help='dataset used for training')
    parser.add_argument('--n-eps', type=int, default=1, help='number of local iterations')
    parser.add_argument('--n_clients', type=int, default=5, help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication round')
    parser.add_argument('--init_seed', type=int, default=51290, help="Random seed")
    parser.add_argument('--beta', type=float, default=0.1,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--concen_loss', type=str, default='uniform_norm', choices=['norm', 'uniform_norm'], help='How to measure the modle difference')
    parser.add_argument('--weight_norm', type=str, default='relu', choices=['sum', 'softmax', 'abs', 'relu', 'sigmoid'], help='How to measure the modle difference')
    parser.add_argument('--difference_measure', type=str, default='all', help='How to measure the model difference')
    parser.add_argument('--complementary_metric', type=str, default='PA', help='How to measure the model complementary')
    parser.add_argument('--matrix_alpha', type=float, default=0.25, help='Hyper-parameter for matrix alpha')
    parser.add_argument('--lam', type=float, default=0.001, help="Hyper-parameter in the objective")
    parser.add_argument('--k_principal', type=float, default=3, help='the dimension of the principal component')
    parser.add_argument('--matrix_beta', type=float, default=1.2, help='Hyper-parameter for matrix beta')
    parser.add_argument('--alpha_bound', type=float, default=1, help='Hyper-parameter to avoid concentration')
    parser.add_argument('--target_dir', type=str, default="/home", help='Hyper-parameter to avoid concentration')
    # attack
    parser.add_argument('--attack_type', type=str, default="inv_grad")
    parser.add_argument('--attack_ratio', type=float, default=0.0)
    parser.add_argument('--ssh', action='store_true',
                        help='whether or not we are executing command via ssh. '
                             'If set to True, we will not print anything to screen and only redirect them to log file')

    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--n-latentdims', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=3, help='number of layers for MLP')
    parser.add_argument('--lr1', type=float, default=0.15, help='Initial learning rate of MLP.')
    parser.add_argument('--lr2', type=float, default=0.005, help='Initial learning rate of Combination.')
    parser.add_argument('--wd1', type=float, default=0.0005, help='Weight decay of MLP.')
    parser.add_argument('--wd2', type=float, default=0, help='Weight decay of Combination.')
    parser.add_argument('--sole', action="store_true", help='if one paramter for one level feature')
    parser.add_argument('--dpC', type=float, default=0.3, help='Dropout rate of Combination.')
    parser.add_argument('--dpM', type=float, default=0.4, help='Dropout rate of MLP.')
    parser.add_argument('--tau', type=float, default=0.95, help='tau.')
    parser.add_argument('--bias', default='none', help='bias.')
    parser.add_argument('--mode', type=str, default='disjoint', choices=['disjoint', 'overlapping'])
    parser.add_argument('--base-path', type=str, default='../datasets')
    parser.add_argument('--plain', action="store_true", help='if plain basis')
    parser.add_argument('--checkpt-path', type=str, default='checkpoints')
    parser.add_argument('--gamma', type=float, default=1, help='gamma')

    args = parser.parse_args()
    cfg = dict()

    cfg["optimizer"] = args.optimizer
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist', 'yahoo_answers'}:
        cfg['classes_size'] = 10
    elif args.dataset == 'cifar100':
        cfg['classes_size'] = 100
    elif args.dataset == 'tinyimagenet':
        cfg['classes_size'] = 200
    cfg['client_num'] = args.n_clients
    cfg['self_wight'] = 'loss'

    if args.dataset == 'Cora':
        if args.mode == 'disjoint':
            if args.n_clients == 5:
                args.k_principal = 3
                args.K = 10
                args.n_latentdims = 32
                args.n_layers = 4
                args.lr1 = 0.149001
                args.lr2 = 0.178501
                args.wd1 = 0.0015
                args.wd2 = 0.002
                args.dpC = 0.87
                args.dpM = 0.05
                args.tau = 0.96
                args.matrix_alpha = 0.0
                args.lam = 0.0
                args.matrix_beta = 1.6
            elif args.n_clients == 10:
                args.gamma = 1.6
                args.K = 7
                args.n_latentdims = 512
                args.n_layers = 3
                args.lr1 = 0.15
                args.lr2 = 0.01
                args.wd1 = 0.0005
                args.wd2 = 0.000005
                args.dpC = 0.3
                args.dpM = 0.15
                args.tau = 0.975
                args.matrix_alpha = 0.25
                args.lam = 0.001
                args.matrix_beta = 2.2
            elif args.n_clients == 20:
                args.gamma = 1.6
                args.K = 7
                args.n_latentdims = 512
                args.n_layers = 3
                args.lr1 = 0.15
                args.lr2 = 0.01
                args.wd1 = 0.0005
                args.wd2 = 0.000005
                args.dpC = 0.3
                args.dpM = 0.15
                args.tau = 0.975
                args.matrix_alpha = 0.25
                args.lam = 0.001
                args.matrix_beta = 2.2
        elif args.mode == 'overlapping':
            if args.n_clients == 10:
                args.k_principal = 1
                args.gamma = 5
                args.K = 7
                args.n_latentdims = 128
                args.n_layers = 3
                args.lr1 = 0.015
                args.lr2 = 0.05
                args.wd1 = 0.5e-6
                args.wd2 = 0.0000005
                args.dpC = 0
                args.dpM = 0.5
                args.tau = 0.985
                args.matrix_alpha = 0.282
                args.lam = 0.01
                args.matrix_beta = 1.5
            elif args.n_clients == 30:
                args.k_principal = 1
                args.gamma = 1
                args.K = 12
                args.n_latentdims = 128
                args.n_layers = 3
                args.lr1 = 0.02
                args.lr2 = 0.02
                args.wd1 = 0.0000005
                args.wd2 = 0.0000005
                args.dpC = 0.3
                args.dpM = 0.25
                args.tau = 0.98
                args.matrix_alpha = 0.28
                args.lam = 0.001
                args.matrix_beta = 2.22
            elif args.n_clients == 50:
                args.k_principal = 1
                args.gamma = 8
                args.K = 7
                args.n_latentdims = 256
                args.n_layers = 3
                args.lr1 = 0.015
                args.lr2 = 0.024
                args.wd1 = 0.5e-6
                args.wd2 = 0.000005
                args.dpC = 0.1
                args.dpM = 0.2
                args.tau = 0.998
                args.matrix_alpha = 0.43
                args.lam = 0.01
                args.matrix_beta = 2.2
    elif args.dataset == 'CiteSeer':
        if args.mode == 'disjoint':
            if args.n_clients == 5:
                args.k_principal = 1
                args.gamma = 3
                args.K = 7
                args.n_latentdims = 128
                args.n_layers = 3
                args.lr1 = 0.15
                args.lr2 = 0.055
                args.wd1 = 0.5e-5
                args.wd2 = 0.000001
                args.dpC = 0
                args.dpM = 0.2
                args.tau = 0.998
                args.matrix_alpha = 0.3
                args.lam = 0.01
                args.matrix_beta = 2.2
            elif args.n_clients == 10:
                args.k_principal = 1
                args.gamma = 3
                args.K = 7
                args.n_latentdims = 128
                args.n_layers = 3
                args.lr1 = 0.15
                args.lr2 = 0.055
                args.wd1 = 0.5e-6
                args.wd2 = 0.000005
                args.dpC = 0
                args.dpM = 0.2
                args.tau = 0.998
                args.matrix_alpha = 0.28
                args.lam = 0.01
                args.matrix_beta = 2.2
            elif args.n_clients == 20:
                args.k_principal = 1
                args.gamma = 3
                args.K = 7
                args.n_latentdims = 128
                args.n_layers = 3
                args.lr1 = 0.15
                args.lr2 = 0.055
                args.wd1 = 0.5e-4
                args.wd2 = 0.000001
                args.dpC = 0
                args.dpM = 0.2
                args.tau = 0.998
                args.matrix_alpha = 0.31
                args.lam = 0.01
                args.matrix_beta = 2.2
        elif args.mode == 'overlapping':
            if args.n_clients == 10:
                args.k_principal = 1
                args.gamma = 3
                args.K = 7
                args.n_latentdims = 128
                args.n_layers = 3
                args.lr1 = 0.15
                args.lr2 = 0.065
                args.wd1 = 0.5e-5
                args.wd2 = 0.0005
                args.dpC = 0
                args.dpM = 0.2
                args.tau = 0.998
                args.matrix_alpha = 0.38
                args.lam = 0.01
                args.matrix_beta = 2.3
            elif args.n_clients == 30:
                args.k_principal = 1
                args.K = 3
                args.n_latentdims = 64
                args.n_layers = 3
                args.lr1 = 0.15
                args.lr2 = 0.207
                args.wd1 = 0.00005
                args.wd2 = 0.0005
                args.dpC = 0.3
                args.dpM = 0.33
                args.tau = 0.978
                args.matrix_alpha = 0.36
                args.lam = 0.001
                args.matrix_beta = 2.4
            elif args.n_clients == 50:
                args.k_principal = 1
                args.gamma = 3
                args.K = 6
                args.n_latentdims = 64
                args.n_layers = 5
                args.lr1 = 0.15
                args.lr2 = 0.15
                args.wd1 = 0.5e-5
                args.wd2 = 0.00005
                args.dpC = 0.31
                args.dpM = 0.3
                args.tau = 0.98
                args.matrix_alpha = 0.3
                args.lam = 0.01
                args.matrix_beta = 2.25
    elif args.dataset == 'PubMed':
        if args.mode == 'disjoint':
            if args.n_clients == 5:
                args.K = 10
                args.n_latentdims = 128
                args.n_layers = 4
                args.lr1 = 0.1
                args.lr2 = 0.05
                args.wd1 = 0.0005
                args.wd2 = 0.0001
                args.dpC = 0.4
                args.dpM = 0
                args.tau = 0.95
                args.matrix_alpha = 0.25
                args.lam = 0.01
                args.matrix_beta = 1.2
            elif args.n_clients == 10:
                args.k_principal = 6
                args.gamma = 14
                args.K = 13
                args.n_latentdims = 128
                args.n_layers = 4
                args.lr1 = 0.1
                args.lr2 = 0.05
                args.wd1 = 0.0005
                args.wd2 = 0.0001
                args.dpC = 0.4
                args.dpM = 0
                args.tau = 0.8
                args.matrix_alpha = 0.25
                args.lam = 0.01
                args.matrix_beta = 1.8
            elif args.n_clients == 20:
                args.k_principal = 6
                args.gamma = 8
                args.K = 13
                args.n_latentdims = 128
                args.n_layers = 4
                args.lr1 = 0.1
                args.lr2 = 0.05
                args.wd1 = 0.0005
                args.wd2 = 0.0001
                args.dpC = 0.4
                args.dpM = 0
                args.tau = 0.8
                args.matrix_alpha = 0.25
                args.lam = 0.01
                args.matrix_beta = 1.75
        elif args.mode == 'overlapping':
            if args.n_clients == 10:
                args.k_principal = 6
                args.gamma = 10
                args.K = 8
                args.n_latentdims = 128
                args.n_layers = 4
                args.lr1 = 0.1
                args.lr2 = 0.05
                args.wd1 = 0.0005
                args.wd2 = 0.0001
                args.dpC = 0.4
                args.dpM = 0
                args.tau = 0.8
                args.matrix_alpha = 0.27
                args.lam = 0.01
                args.matrix_beta = 1.75
            elif args.n_clients == 30:
                args.k_principal = 6
                args.gamma = 10
                args.K = 4
                args.n_latentdims = 128
                args.n_layers = 4
                args.lr1 = 0.1
                args.lr2 = 0.05
                args.wd1 = 0.00005
                args.wd2 = 0.00005
                args.dpC = 0.3
                args.dpM = 0.08
                args.tau = 0.92
                args.matrix_alpha = 0.27
                args.lam = 0.01
                args.matrix_beta = 1.75
            elif args.n_clients == 50:
                args.k_principal = 6
                args.gamma = 10
                args.K = 8
                args.n_latentdims = 128
                args.n_layers = 4
                args.lr1 = 0.1
                args.lr2 = 0.05
                args.wd1 = 0.0005
                args.wd2 = 0.0001
                args.dpC = 0.4
                args.dpM = 0
                args.tau = 0.8
                args.matrix_alpha = 0.1
                args.lam = 0.01
                args.matrix_beta = 1.75
    elif args.dataset == 'Computers':
        if args.mode == 'disjoint':
            if args.n_clients == 5:
                args.k_principal = 3
                args.gamma = 3
                args.K = 12
                args.n_latentdims = 128
                args.n_layers = 4
                args.lr1 = 0.02
                args.lr2 = 0.025
                args.wd1 = 0.00005
                args.wd2 = 0.000005
                args.dpC = 0.3
                args.dpM = 0.3
                args.tau = 0.97
                args.matrix_alpha = 0.16
                args.lam = 0.001
                args.matrix_beta = 2.2
                args.comm_round = 200
                args.n_eps = 3
            elif args.n_clients == 10:
                args.k_principal = 1
                args.gamma = 5
                args.K = 7
                args.n_latentdims = 128
                args.n_layers = 3
                args.lr1 = 0.0478
                args.lr2 = 0.045
                args.wd1 = 0.0000005
                args.wd2 = 0.000005
                args.dpC = 0.3
                args.dpM = 0.1
                args.tau = 0.98
                args.matrix_alpha = 0.1
                args.lam = 0.001
                args.matrix_beta = 2.25
                args.comm_round = 200
                args.n_eps = 3
            elif args.n_clients == 20:
                args.k_principal = 1
                args.gamma = 4.5
                args.K = 7
                args.n_latentdims = 128
                args.n_layers = 3
                args.lr1 = 0.0478
                args.lr2 = 0.045
                args.wd1 = 0.0000005
                args.wd2 = 0.0000005
                args.dpC = 0.3
                args.dpM = 0.1
                args.tau = 0.992
                args.matrix_alpha = 0.1
                args.lam = 0.001
                args.matrix_beta = 2.3
                args.comm_round = 200
                args.n_eps = 3
        elif args.mode == 'overlapping':
            if args.n_clients == 10:
                args.k_principal = 8
                args.gamma = 10
                args.K = 7
                args.n_latentdims = 128
                args.n_layers = 3
                args.lr1 = 0.035
                args.lr2 = 0.025
                args.wd1 = 5e-6
                args.wd2 = 0.00005
                args.dpC = 0.3
                args.dpM = 0.3
                args.tau = 0.991111
                args.matrix_alpha = 0.0466
                args.lam = 0.001
                args.matrix_beta = 2.2
                args.comm_round = 200
                args.n_eps = 3
            elif args.n_clients == 30:
                args.k_principal = 8
                args.gamma = 10
                args.K = 7
                args.n_latentdims = 128
                args.n_layers = 3
                args.lr1 = 0.015
                args.lr2 = 0.015
                args.wd1 = 5e-6
                args.wd2 = 0.00005
                args.dpC = 0.3
                args.dpM = 0.15
                args.tau = 0.98
                args.matrix_alpha = 0.04
                args.lam = 0.001
                args.matrix_beta = 2.5
                args.comm_round = 200
                args.n_eps = 3
            elif args.n_clients == 50:
                args.k_principal = 1
                args.gamma = 10
                args.K = 5
                args.n_latentdims = 128
                args.n_layers = 3
                args.lr1 = 0.015
                args.lr2 = 0.02
                args.wd1 = 0.5e-6
                args.wd2 = 0.000005
                args.dpC = 0.325
                args.dpM = 0.125
                args.tau = 0.9942
                args.matrix_alpha = 0.254
                args.lam = 0.001
                args.matrix_beta = 2.5
                args.comm_round = 200
                args.n_eps = 3
    elif args.dataset == 'Photo':
        if args.mode == 'disjoint':
            if args.n_clients == 5:
                args.K = 10
                args.n_latentdims = 256
                args.n_layers = 5
                args.lr1 = 0.006
                args.lr2 = 0.005
                args.wd1 = 0.00005
                args.wd2 = 0.0005
                args.dpC = 0.25
                args.dpM = 0.4
                args.tau = 0.95
                args.matrix_alpha = 0.25
                args.lam = 0.01
                args.matrix_beta = 1.2
                args.comm_round = 200
                args.n_eps = 2
            elif args.n_clients == 10:
                args.gamma = 5
                args.K = 12
                args.n_latentdims = 256
                args.n_layers = 5
                args.lr1 = 0.016
                args.lr2 = 0.015
                args.wd1 = 0.00005
                args.wd2 = 0.0005
                args.dpC = 0.25
                args.dpM = 0.3
                args.tau = 0.955
                args.matrix_alpha = 0.28
                args.lam = 0.01
                args.matrix_beta = 2.2
                args.comm_round = 200
                args.n_eps = 2
            elif args.n_clients == 20:
                args.K = 10
                args.n_latentdims = 256
                args.n_layers = 5
                args.lr1 = 0.006
                args.lr2 = 0.005
                args.wd1 = 0.000001
                args.wd2 = 0.0000001
                args.dpC = 0.25
                args.dpM = 0.3
                args.tau = 0.99
                args.matrix_alpha = 0.23
                args.lam = 0.01
                args.matrix_beta = 1.25
                args.comm_round = 200
                args.n_eps = 2
        elif args.mode == 'overlapping':
            if args.n_clients == 10:
                args.k_principal = 1
                args.K = 16
                args.n_latentdims = 256
                args.n_layers = 5
                args.lr1 = 0.01
                args.lr2 = 0.01
                args.wd1 = 0.0005
                args.wd2 = 0.005
                args.dpC = 0.25
                args.dpM = 0.25
                args.tau = 0.994
                args.matrix_alpha = 0.12
                args.lam = 0.01
                args.matrix_beta = 2.2
                args.comm_round = 200
                args.n_eps = 2
            elif args.n_clients == 30:
                args.gamma = 6
                args.k_principal = 1
                args.K = 7
                args.n_latentdims = 256
                args.n_layers = 5
                args.lr1 = 0.01
                args.lr2 = 0.01
                args.wd1 = 0.000005
                args.wd2 = 0.000005
                args.dpC = 0.25
                args.dpM = 0.2
                args.tau = 0.9945
                args.matrix_alpha = 0.156
                args.lam = 0.01
                args.matrix_beta = 2.2
                args.comm_round = 200
                args.n_eps = 2
            elif args.n_clients == 50:
                args.gamma = 3
                args.k_principal = 1
                args.K = 7
                args.n_latentdims = 256
                args.n_layers = 5
                args.lr1 = 0.012
                args.lr2 = 0.0112
                args.wd1 = 0.000001
                args.wd2 = 0.000001
                args.dpC = 0.25
                args.dpM = 0.25
                args.tau = 0.991
                args.matrix_alpha = 0.16
                args.lam = 0.01
                args.matrix_beta = 2.24
                args.comm_round = 200
                args.n_eps = 2
    elif args.dataset == 'ogbn-arxiv':
        if args.mode == 'disjoint':
            if args.n_clients == 5:
                args.k_principal = 1
                args.gamma = 1
                args.K = 14
                args.n_latentdims = 1024
                args.n_layers = 3
                args.lr1 = 0.01
                args.lr2 = 0.06
                args.wd1 = 0.5e-4
                args.wd2 = 0.005
                args.dpC = 0.3
                args.dpM = 0.1
                args.tau = 0.97
                args.matrix_alpha = 0.17
                args.lam = 0.001
                args.matrix_beta = 2.2
                args.comm_round = 200
                args.n_eps = 2
            elif args.n_clients == 10:
                args.k_principal = 1
                args.gamma = 1
                args.K = 15
                args.n_latentdims = 1024
                args.n_layers = 3
                args.lr1 = 0.01
                args.lr2 = 0.05
                args.wd1 = 0.5e-4
                args.wd2 = 0.005
                args.dpC = 0.3
                args.dpM = 0.1
                args.tau = 0.97
                args.matrix_alpha = 0.17
                args.lam = 0.001
                args.matrix_beta = 2.2
                args.comm_round = 200
                args.n_eps = 2
            elif args.n_clients == 20:
                args.k_principal = 1
                args.gamma = 2
                args.K = 15
                args.n_latentdims = 1024
                args.n_layers = 3
                args.lr1 = 0.025
                args.lr2 = 0.055
                args.wd1 = 5e-6
                args.wd2 = 0.005
                args.dpC = 0.3
                args.dpM = 0.3
                args.tau = 0.96
                args.matrix_alpha = 0.17
                args.lam = 0.001
                args.matrix_beta = 2.2
                args.comm_round = 200
                args.n_eps = 2
        elif args.mode == 'overlapping':
            if args.n_clients == 10:
                args.k_principal = 1
                args.K = 13
                args.n_latentdims = 512
                args.n_layers = 3
                args.lr1 = 0.01
                args.lr2 = 0.05
                args.wd1 = 0.5e-4
                args.wd2 = 0.005
                args.dpC = 0.3
                args.dpM = 0.1
                args.tau = 0.97
                args.matrix_alpha = 0.17
                args.lam = 0.001
                args.matrix_beta = 2.22
                args.comm_round = 200
                args.n_eps = 2
            elif args.n_clients == 30:
                args.k_principal = 1
                args.K = 13
                args.n_latentdims = 512
                args.n_layers = 3
                args.lr1 = 0.01
                args.lr2 = 0.04
                args.wd1 = 0.5e-4
                args.wd2 = 0.005
                args.dpC = 0.2
                args.dpM = 0.05
                args.tau = 0.98
                args.matrix_alpha = 0.19
                args.lam = 0.001
                args.matrix_beta = 2.28
                args.comm_round = 200
                args.n_eps = 2
            elif args.n_clients == 50:
                args.k_principal = 1
                args.gamma = 10
                args.K = 13
                args.n_latentdims = 256
                args.n_layers = 3
                args.lr1 = 0.015
                args.lr2 = 0.05
                args.wd1 = 0.5e-6
                args.wd2 = 0.00005
                args.dpC = 0.2
                args.dpM = 0.05
                args.tau = 0.97
                args.matrix_alpha = 0.27
                args.lam = 0.001
                args.matrix_beta = 2.32
                args.comm_round = 200
                args.n_eps = 2
    elif args.dataset == 'Roman-empire':
        if args.mode == 'disjoint':
            if args.n_clients == 5:
                args.gamma = 1.6
                args.K = 12
                args.n_latentdims = 512
                args.n_layers = 3
                args.lr1 = 0.01
                args.lr2 = 0.05
                args.wd1 = 0.5e-5
                args.wd2 = 0.005
                args.dpC = 0.3
                args.dpM = 0.3
                args.dpC = 0.3
                args.dpM = 0.3
                args.tau = 0.7
                args.matrix_alpha = 0.12
                args.lam = 0.001
                args.matrix_beta = 1.3
            elif args.n_clients == 10:
                args.k_principal = 9
                args.gamma = 4
                args.K = 13
                args.n_latentdims = 256
                args.n_layers = 3
                args.lr1 = 0.01
                args.lr2 = 0.05
                args.wd1 = 0.5e-5
                args.wd2 = 0.0005
                args.dpC = 0
                args.dpM = 0.6
                args.tau = 0.7
                args.matrix_alpha = 0.25
                args.lam = 0.01
                args.matrix_beta = 1.2
            elif args.n_clients == 20:
                args.k_principal = 9
                args.gamma = 4
                args.K = 13
                args.n_latentdims = 256
                args.n_layers = 3
                args.lr1 = 0.01
                args.lr2 = 0.05
                args.wd1 = 0.5e-5
                args.wd2 = 0.0005
                args.dpC = 0
                args.dpM = 0.6
                args.tau = 0.7
                args.matrix_alpha = 0.25
                args.lam = 0.01
                args.matrix_beta = 1.2
        elif args.mode == 'overlapping':
            if args.n_clients == 10:
                args.k_principal = 9
                args.gamma = 4
                args.K = 7
                args.n_latentdims = 256
                args.n_layers = 3
                args.lr1 = 0.015
                args.lr2 = 0.05
                args.wd1 = 0.5e-5
                args.wd2 = 0.0005
                args.dpC = 0
                args.dpM = 0.48
                args.tau = 0.7
                args.matrix_alpha = 0.25
                args.lam = 0.01
                args.matrix_beta = 1.2
            elif args.n_clients == 30:
                args.k_principal = 9
                args.gamma = 10
                args.K = 7
                args.n_latentdims = 256
                args.n_layers = 3
                args.lr1 = 0.0155834
                args.lr2 = 0.05
                args.wd1 = 0.5e-5
                args.wd2 = 0.0005
                args.dpC = 0
                args.dpM = 0.48
                args.tau = 0.7
                args.matrix_alpha = 0.25
                args.lam = 0.01
                args.matrix_beta = 2.2
            elif args.n_clients == 50:
                args.k_principal = 9
                args.gamma = 10
                args.K = 7
                args.n_latentdims = 256
                args.n_layers = 3
                args.lr1 = 0.023
                args.lr2 = 0.05
                args.wd1 = 0.5e-5
                args.wd2 = 0.0005
                args.dpC = 0
                args.dpM = 0.48
                args.tau = 0.7
                args.matrix_alpha = 0.22
                args.lam = 0.01
                args.matrix_beta = 1.2
    elif args.dataset == 'Amazon-ratings':
        if args.mode == 'disjoint':
            if args.n_clients == 5:
                args.k_principal = 7
                args.gamma = 4
                args.K = 10
                args.n_latentdims = 128
                args.n_layers = 3
                args.lr1 = 0.015
                args.lr2 = 0.025
                args.wd1 = 1e-6
                args.wd2 = 1e-6
                args.dpC = 0.35
                args.dpM = 0.17
                args.tau = 0.6
                args.matrix_alpha = 0.17
                args.lam = 0.05
                args.matrix_beta = 2.4
            elif args.n_clients == 10:
                args.k_principal = 7
                args.gamma = 4
                args.K = 13
                args.n_latentdims = 128
                args.n_layers = 3
                args.lr1 = 0.015
                args.lr2 = 0.025
                args.wd1 = 5e-6
                args.wd2 = 5e-6
                args.dpC = 0.35
                args.dpM = 0.18
                args.tau = 0.6
    
                args.matrix_alpha = 0.35
                args.lam = 0.05
                args.matrix_beta = 2.4
            elif args.n_clients == 20:
                args.k_principal = 5
                args.gamma = 4
                args.K = 11
                args.n_latentdims = 128
                args.n_layers = 3
                args.lr1 = 0.015
                args.lr2 = 0.025
                args.wd1 = 1e-6
                args.wd2 = 1e-6
                args.dpC = 0.35
                args.dpM = 0.35
                args.tau = 0.6
                args.matrix_alpha = 0.19
                args.lam = 0.05
                args.matrix_beta = 2.33
        elif args.mode == 'overlapping':
            if args.n_clients == 10:
                args.k_principal = 8
                args.gamma = 4
                args.K = 12
                args.n_latentdims = 128
                args.n_layers = 3
                args.lr1 = 0.015
                args.lr2 = 0.025
                args.wd1 = 1e-6
                args.wd2 = 1e-6
                args.dpC = 0.25
                args.dpM = 0.25
                args.tau = 0.6
                args.matrix_alpha = 0.4
                args.lam = 0.05
                args.matrix_beta = 2.4
            elif args.n_clients == 30:
                args.k_principal = 7
                args.gamma = 4
                args.K = 11
                args.n_latentdims = 128
                args.n_layers = 3
                args.lr1 = 0.05
                args.lr2 = 0.05
                args.wd1 = 5e-6
                args.wd2 = 5e-6
                args.dpC = 0.35
                args.dpM = 0.35
                args.tau = 0.6
                args.matrix_alpha = 0.17
                args.lam = 0.05
                args.matrix_beta = 2.2
            elif args.n_clients == 50:
                args.k_principal = 7
                args.gamma = 4
                args.K = 11
                args.n_latentdims = 256
                args.n_layers = 3
                args.lr1 = 0.025
                args.lr2 = 0.025
                args.wd1 = 1e-6
                args.wd2 = 1e-6
                args.dpC = 0.35
                args.dpM = 0.35
                args.tau = 0.6
                args.matrix_alpha = 0.17
                args.lam = 0.05
                args.matrix_beta = 2.2
    elif args.dataset == 'Minesweeper':
        if args.mode == 'disjoint':
            args.gamma = 50
            args.K = 10
            args.n_latentdims = 512
            args.n_layers = 3
            args.lr1 = 0.01
            args.lr2 = 0.05
            args.wd1 = 0.5e-4
            args.wd2 = 0.005
            args.dpC = 0.3
            args.dpM = 0.3
            args.tau = 0.7
            args.matrix_alpha = 0.05
            args.lam = 0.001
            args.matrix_beta = 2
        elif args.mode == 'overlapping':
            if args.n_clients == 10:
                args.gamma = 50
                args.K = 10
                args.n_latentdims = 512
                args.n_layers = 3
                args.lr1 = 0.01
                args.lr2 = 0.05
                args.wd1 = 0.5e-5
                args.wd2 = 0.005
                args.dpC = 0.3
                args.dpM = 0.35
                args.tau = 0.65
                args.matrix_alpha = 0.19
                args.lam = 0.001
                args.matrix_beta = 2
            elif args.n_clients == 30:
                args.gamma = 50
                args.K = 10
                args.n_latentdims = 512
                args.n_layers = 3
                args.lr1 = 0.01
                args.lr2 = 0.05
                args.wd1 = 0.5e-4
                args.wd2 = 0.005
                args.dpC = 0.3
                args.dpM = 0.3
                args.tau = 0.7
                args.matrix_alpha = 0.05
                args.lam = 0.001
                args.matrix_beta = 2
            elif args.n_clients == 50:
                args.gamma = 50
                args.K = 10
                args.n_latentdims = 512
                args.n_layers = 3
                args.lr1 = 0.01
                args.lr2 = 0.05
                args.wd1 = 0.5e-4
                args.wd2 = 0.005
                args.dpC = 0.3
                args.dpM = 0.3
                args.tau = 0.65
                args.matrix_alpha = 0.19
                args.lam = 0.001
                args.matrix_beta = 2
    elif args.dataset == 'Tolokers':
        if args.mode == 'disjoint':
            if args.n_clients == 5:
                args.gamma = 1.5
                args.K = 10
                args.n_latentdims = 512
                args.n_layers = 3
                args.lr1 = 0.025
                args.lr2 = 0.05
                args.wd1 = 0.5e-6
                args.wd2 = 0.000005
                args.dpC = 0.3
                args.dpM = 0.3
                args.tau = 0.7
                args.matrix_alpha = 0.28
                args.lam = 0.001
                args.matrix_beta = 1.2
            elif args.n_clients == 10:
                args.gamma = 1.6
                args.K = 10
                args.n_latentdims = 512
                args.n_layers = 3
                args.lr1 = 0.015
                args.lr2 = 0.05
                args.wd1 = 0.5e-5
                args.wd2 = 0.005
                args.dpC = 0.3
                args.dpM = 0.3
                args.tau = 0.7
                args.matrix_alpha = 0.2
                args.lam = 0.001
                args.matrix_beta = 1.2
            elif args.n_clients == 20:
                args.gamma = 10
                args.K = 12
                args.n_latentdims = 512
                args.n_layers = 3
                args.lr1 = 0.4
                args.lr2 = 0.35
                args.wd1 = 0.5e-2
                args.wd2 = 0.05
                args.dpC = 0.3
                args.dpM = 0.28
                args.tau = 0.53
                args.matrix_alpha = 0.0000001
                args.lam = 0.001
                args.matrix_beta = 2.22
        elif args.mode == 'overlapping':
            if args.n_clients == 10:
                args.gamma = 2
                args.K = 8
                args.n_latentdims = 512
                args.n_layers = 3
                args.lr1 = 0.015
                args.lr2 = 0.05
                args.wd1 = 0.5e-5
                args.wd2 = 0.005
                args.dpC = 0.3
                args.dpM = 0.18
                args.tau = 0.7
                args.matrix_alpha = 0.12
                args.lam = 0.001
                args.matrix_beta = 1.2
            elif args.n_clients == 30:
                args.gamma = 4
                args.K = 10
                args.n_latentdims = 512
                args.n_layers = 3
                args.lr1 = 0.025
                args.lr2 = 0.025
                args.wd1 = 5e-6
                args.wd2 = 0.005
                args.dpC = 0.3
                args.dpM = 0.35
                args.tau = 0.62
                args.matrix_alpha = 0.25
                args.lam = 0.001
                args.matrix_beta = 1.2
            elif args.n_clients == 50:
                args.gamma = 8
                args.K = 13
                args.n_latentdims = 256
                args.n_layers = 3
                args.lr1 = 0.15
                args.lr2 = 0.015
                args.wd1 = 0.5e-5
                args.wd2 = 0.001
                args.dpC = 0.26
                args.dpM = 0.26
                args.tau = 0.6144
                args.matrix_alpha = 0.045423
                args.lam = 0.001
                args.matrix_beta = 1.2
    elif args.dataset == 'Questions':
        if args.mode == 'disjoint':
            if args.n_clients == 5:
                args.k_principal = 9
                args.gamma = 4
                args.K = 13
                args.n_latentdims = 256
                args.n_layers = 6
                args.lr1 = 0.01
                args.lr2 = 0.05
                args.wd1 = 0.5e-5
                args.wd2 = 0.0005
                args.dpC = 0
                args.dpM = 0.6
                args.tau = 0.7
                args.matrix_alpha = 0.25
                args.lam = 0.01
                args.matrix_beta = 2.5
            elif args.n_clients == 10:
                args.k_principal = 9
                args.gamma = 4
                args.K = 13
                args.n_latentdims = 256
                args.n_layers = 6
                args.lr1 = 0.01
                args.lr2 = 0.05
                args.wd1 = 0.5e-5
                args.wd2 = 0.0005
                args.dpC = 0
                args.dpM = 0.6
                args.tau = 0.7
                args.matrix_alpha = 0.3
                args.lam = 0.01
                args.matrix_beta = 2.3
            elif args.n_clients == 20:
                args.k_principal = 9
                args.gamma = 4
                args.K = 11
                args.n_latentdims = 256
                args.n_layers = 6
                args.lr1 = 0.01
                args.lr2 = 0.05
                args.wd1 = 0.5e-5
                args.wd2 = 0.0005
                args.dpC = 0
                args.dpM = 0.6
                args.tau = 0.7
                args.matrix_alpha = 0.3
                args.lam = 0.01
                args.matrix_beta = 2
        elif args.mode == 'overlapping':
            if args.n_clients == 10:
                args.k_principal = 3
                args.gamma = 4
                args.K = 7
                args.n_latentdims = 256
                args.n_layers = 3
                args.lr1 = 0.05
                args.lr2 = 0.032
                args.wd1 = 0.5e-6
                args.wd2 = 0.0000005
                args.dpC = 0.3
                args.dpM = 0.3
                args.tau = 0.991
                args.matrix_alpha = 0.28
                args.lam = 0.01
                args.matrix_beta = 2.3
            elif args.n_clients == 30:
                args.k_principal = 9
                args.gamma = 10
                args.K = 6
                args.n_latentdims = 256
                args.n_layers = 6
                args.lr1 = 0.0123
                args.lr2 = 0.05
                args.wd1 = 0.5e-5
                args.wd2 = 0.0005
                args.dpC = 0
                args.dpM = 0.125
                args.tau = 0.9945
                args.matrix_alpha = 0.28
                args.lam = 0.01
                args.matrix_beta = 2.2
            elif args.n_clients == 50:
                args.k_principal = 3
                args.gamma = 4
                args.K = 11
                args.n_latentdims = 256
                args.n_layers = 6
                args.lr1 = 0.05
                args.lr2 = 0.025
                args.wd1 = 0.5e-6
                args.wd2 = 0.0005
                args.dpC = 0.3
                args.dpM = 0.3
                args.tau = 0.7
                args.matrix_alpha = 0.17
                args.lam = 0.01
                args.matrix_beta = 2.2

    cfg["comm_round"] = args.comm_round
    return args , cfg