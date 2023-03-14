"""
utility functions and global variables
"""

import os
from distutils.util import strtobool

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

DEFAULT_DIC = {'sample_size': None, 'dataset_name': 'Cora', 'num_hops': 2, 'max_dist': 10, 'max_nodes_per_hop': 10,
               'data_appendix': None, 'val_pct': 0.1, 'test_pct': 0.2, 'train_sample': 1, 'dynamic_train': True,
               'dynamic_val': True, 'model': 'hashing', 'dbname': None, 'sign_k': 2,
               'dynamic_test': True, 'node_label': 'drnl', 'ratio_per_hop': 1, 'use_feature': True, 'dropout': 0,
               'label_dropout': 0, 'feature_dropout': 0, 'label_features_branch': True, 'use_bn': True,
               'add_normed_features': False, 'use_RA': False, 'hidden_channels': 32, 'load_features': True,
               'load_hashes': True, 'use_zero_one': True, 'hash_db': None, 'test_citation_sample_size': None,
               'val_citation_sample_size': None, 'wandb': False, 'batch_size': 32, 'num_workers': 1,
               'cache_train_structure_features': False, 'cache_val_structure_features': False,
               'cache_test_structure_features': False,
               'citation_sample_size': None, 'eval_batch_size': 1000, 'bulk_train': False, 'num_negs': 1}


def print_model_params(model):
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data.shape)


def select_embedding(args, num_nodes, device):
    """
    select a node embedding. Used by SEAL models (the E is Embedding)
    and needed for ogb-ddi where there are no node features
    :param args: Namespace of cmd args
    :param num_nodes: Int number of nodes to produce embeddings for
    :param device: cpu or cuda
    :return: Torch.nn.Embedding [n_nodes, args.hidden_channels]
    """
    if args.train_node_embedding:
        emb = torch.nn.Embedding(num_nodes, args.hidden_channels).to(device)
    elif args.pretrained_node_embedding:
        weight = torch.load(args.pretrained_node_embedding)
        emb = torch.nn.Embedding.from_pretrained(weight)
        emb.weight.requires_grad = False
    else:
        emb = None
    return emb


def neighbors(fringe, A, outgoing=True):
    """
    Retrieve neighbours of nodes within the fringe
    :param fringe: set of node IDs
    :param A: scipy CSR sparse adjacency matrix
    :param outgoing: bool
    :return:
    """
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def get_src_dst_degree(src, dst, A, max_nodes):
    """
    Assumes undirected, unweighted graph
    :param src: Int Tensor[edges]
    :param dst: Int Tensor[edges]
    :param A: scipy CSR adjacency matrix
    :param max_nodes: cap on max node degree
    :return:
    """
    src_degree = A[src].sum() if (max_nodes is None or A[src].sum() <= max_nodes) else max_nodes
    dst_degree = A[dst].sum() if (max_nodes is None or A[src].sum() <= max_nodes) else max_nodes
    return src_degree, dst_degree


def str2bool(x):
    """
    hack to allow wandb to tune boolean cmd args
    :param x: str of bool
    :return: bool
    """
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f'Unrecognised type {type(x)}')
