import argparse

def load_arg_parser():
    parser = argparse.ArgumentParser()
    parser = add_base_args(parser)
    parser = add_gnn_args(parser)
    args = parser.parse_args()

    return args

def add_base_args(parser):
    parser.add_argument('--gnn', type=str, default='GCN', choices=['GCN', 'GAT'])
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=2020)

    return parser

def add_gnn_args(parser):
    # GCN
    # no args

    # GAT
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--merge', type=str, default='cat', choices=['cat', 'mean'])

    return parser