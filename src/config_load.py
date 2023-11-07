import os
import argparse
from yacs.config import CfgNode as CN
from src.utils import str2bool
from math import inf
# adopted from torch_geometric.graphgym.config
import yaml 
from typing import Dict, Any
from torch_geometric.graphgym.config import (cfg, dump_cfg, set_cfg, load_cfg,
                                             makedirs_rm_exist)

import argparse

def dict_to_argparser(dictionary):
    parser = argparse.ArgumentParser(description='Command line arguments from dictionary')

    for key, value in dictionary.items():
        # Determine the type of value and set appropriate argument type
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', action='store_true', help=f'Set {key} to True')
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value, help=f'Set {key} (default: {value})')

    return parser

# Create an argparse parser from the dictionary



def recursive_function(data, parent_parser=None):
    parser = parent_parser if parent_parser else {}

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                recursive_function(value, parent_parser=parser)
            else:
                # Add arguments based on the dictionary keys and values
                parser.add_argument(f'--{key}', type=type(value), default=value)

    return parser


def cfgnode_to_dict(cfg_node):
    cfg_dict = {}
    for key, value in cfg_node.items():
        if isinstance(value, CN):
            value = cfgnode_to_dict(value)  # Recursively convert nested CfgNodes
        cfg_dict[key] = value
    return cfg_dict


def merge_from_other_cfg(cfg, config):
    cfg_dict = cfgnode_to_dict(cfg)
    config_dict = cfgnode_to_dict(config)
    merged_dict = merge_dicts(config_dict, cfg_dict)
    
    merged_from_other_cfg = CN(merged_dict)
    return merged_from_other_cfg

def merge_dicts(dict1, dict2):
    result = dict(dict1)  # Create a copy of the first dictionary

    for key, value2 in dict2.items():
        if key in result:
            value1 = result[key]
            if isinstance(value1, dict) and isinstance(value2, dict):
                # If both values are dictionaries, recursively merge them
                result[key] = merge_dicts(value1, value2)
            else:
                # Choose the one with more information (non-None, non-empty)
                if value1 is not None and (not isinstance(value1, dict) or value1):
                    result[key] = value1
                else:
                    result[key] = value2
        else:
            result[key] = value2

    return result

def set_cfg(cfg):
    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Cuda device number, used for machine with multiple gpus
    cfg.device = 0
    # Whether fix the running seed to remove randomness
    cfg.seed = 1
    # Number of runs with random init
    cfg.runs = 4
    cfg.gnn = CN()
        
    cfg.lm = CN()

    # ------------------------------------------------------------------------ #
    # GNN Model options
    # ------------------------------------------------------------------------ #
    cfg.gnn.model = CN()
    # GNN model name
    cfg.gnn.model.name = 'ELPH'
    # Number of gnn layers
    # cfg.gnn.model.num_layers = 4
    # # Hidden size of the model
    # cfg.gnn.model.hidden_dim = 128

    # ------------------------------------------------------------------------ #
    # GNN Training options
    # ------------------------------------------------------------------------ #
    cfg.gnn.train = CN()
    # Use PyG or DGL
    cfg.gnn.train.use_dgl = False
    # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
    cfg.gnn.train.weight_decay = 0.0
    # Maximal number of epochs
    cfg.gnn.train.epochs = 200
    # Node feature type, options: ogb, TA, P, E
    cfg.gnn.train.feature_type = 'TA_P_E'
    # Number of epochs with no improvement after which training will be stopped
    cfg.gnn.train.early_stop = 50
    # Base learning rate
    cfg.gnn.train.lr = 0.01
    # L2 regularization, weight decay
    cfg.gnn.train.wd = 0.0
    # Dropout rate
    cfg.gnn.train.dropout = 0.0

    # ------------------------------------------------------------------------ #
    # LM Model options
    # ------------------------------------------------------------------------ #
    cfg.lm.model = CN()
    # LM model name
    cfg.lm.model.name = 'microsoft/deberta-base'
    cfg.lm.model.feat_shrink = ""

    # ------------------------------------------------------------------------ #
    # LM Training options
    # ------------------------------------------------------------------------ #
    cfg.lm.train = CN()
    #  Number of samples computed once per batch per device
    cfg.lm.train.batch_size = 9
    # Number of training steps for which the gradients should be accumulated
    cfg.lm.train.grad_acc_steps = 1
    # Base learning rate
    cfg.lm.train.lr = 2e-5
    # Maximal number of epochs
    cfg.lm.train.epochs = 0.05
    # The number of warmup steps
    cfg.lm.train.warmup_epochs = 0.6
    # Number of update steps between two evaluations
    cfg.lm.train.eval_patience = 50000
    # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
    cfg.lm.train.weight_decay = 0.0
    # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
    cfg.lm.train.dropout = 0.3
    # The dropout ratio for the attention probabilities
    cfg.lm.train.att_dropout = 0.1
    # The dropout ratio for the classifier
    cfg.lm.train.cla_dropout = 0.4
    # Whether or not to use the gpt responses (i.e., explanation and prediction) as text input
    # If not, use the original text attributes (i.e., title and abstract)
    cfg.lm.train.use_gpt = False

    # Cuda device number, used for machine with multiple gpus
    cfg.device = 0
    # Whether fix the running seed to remove randomness
    cfg.seed = 1

    cfg.dataset = CN()
    # root of dataset 
    cfg.dataset.name = 'cora'
    cfg.dataset.feature_type = 'TA'
    cfg.dataset.use_feature = False
    cfg.dataset.use_text = ""

    return cfg

# Principle means that if an option is defined in a YACS config object,
# then your program should set that configuration option using cfg.merge_from_list(opts) and not by defining,
# for example, --train-scales as a command line argument that is then used to set cfg.TRAIN.SCALES.

def recursive_search_key(data, target_key, results=None, current_key=None):
    if results is None:
        results = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_key = key if current_key is None else f"{current_key}.{key}"
            if key == target_key:
                results.append(new_key)
            if isinstance(value, (dict, list)):
                recursive_search_key(value, target_key, results, new_key)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            new_key = current_key if current_key is not None else str(index)
            recursive_search_key(item, target_key, results, new_key)

    return results



def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser(description='Efficient Link Prediction with Hashes (ELPH)')
    # parser.add_argument('--dataset_name', type=str, required=True,
    #                     choices=['cora', 'Citeseer', 'pubmed', 'ogbl-ppa', 'ogbl-collab', 'ogbl-ddi', 'ogbn-products',
    #                              'ogbl-citation2', 'ogbn-arxiv'])
    # parser.add_argument('--use_text', type=str2bool, required=True,
    #                     help='whether to use text features')
    
    # parser.add_argument('--val_pct', type=float, default=0.1,
    #                     help='the percentage of supervision edges to be used for validation. These edges will not appear'
    #                          ' in the training set and will only be used as message passing edges in the test set')
    # parser.add_argument('--test_pct', type=float, default=0.2,
    #                     help='the percentage of supervision edges to be used for test. These edges will not appear'
    #                          ' in the training or validation sets for either supervision or message passing')
    # parser.add_argument('--train_samples', type=float, default=inf, help='the number of training edges or % if < 1')
    # parser.add_argument('--val_samples', type=float, default=inf, help='the number of val edges or % if < 1')
    # parser.add_argument('--test_samples', type=float, default=inf, help='the number of test edges or % if < 1')
    # parser.add_argument('--preprocessing', type=str, default=None)
    # parser.add_argument('--sign_k', type=int, default=0)
    # parser.add_argument('--load_features', action='store_true', help='load node features from disk')
    # parser.add_argument('--load_hashes', action='store_true', help='load hashes from disk')
    # parser.add_argument('--cache_subgraph_features', action='store_true',
    #                     help='write / read subgraph features from disk')
    # parser.add_argument('--train_cache_size', type=int, default=inf, help='the number of training edges to cache')
    # parser.add_argument('--year', type=int, default=0, help='filter training data from before this year')
    # # GNN settings
    # parser.add_argument('--model', type=str, default='BUDDY')
    # parser.add_argument('--hidden_channels', type=int, default=1024)
    # parser.add_argument('--batch_size', type=int, default=1024)
    # parser.add_argument('--eval_batch_size', type=int, default=1000000,
    #                     help='eval batch size should be largest the GPU memory can take - the same is not necessarily true at training time')
    # parser.add_argument('--label_dropout', type=float, default=0.5)
    # parser.add_argument('--feature_dropout', type=float, default=0.5)
    # parser.add_argument('--sign_dropout', type=float, default=0.5)
    # parser.add_argument('--save_model', action='store_true', help='save the model to use later for inference')
    # parser.add_argument('--feature_prop', type=str, default='gcn',
    #                     help='how to propagate ELPH node features. Values are gcn, residual (resGCN) or cat (jumping knowledge networks)')
    # # SEAL settings
    # parser.add_argument('--dropout', type=float, default=0.5)
    # parser.add_argument('--num_seal_layers', type=int, default=3)
    # parser.add_argument('--sortpool_k', type=float, default=0.6)
    # parser.add_argument('--label_pooling', type=str, default='add', help='add or mean')
    # parser.add_argument('--seal_pooling', type=str, default='edge', help='how SEAL pools features in the subgraph')
    # # Subgraph settings
    # parser.add_argument('--num_hops', type=int, default=1)
    # parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    # parser.add_argument('--max_nodes_per_hop', type=int, default=None)
    # parser.add_argument('--node_label', type=str, default='drnl')
    # parser.add_argument('--max_dist', type=int, default=4)
    # parser.add_argument('--max_z', type=int, default=1000,
    #                     help='the size of the label embedding table. ie. the maximum number of labels possible')
    # parser.add_argument('--use_feature', type=str2bool, default=True,
    #                     help="whether to use raw node features as GNN input")
    # parser.add_argument('--use_struct_feature', type=str2bool, default=True,
    #                     help="whether to use structural graph features as GNN input")
    # parser.add_argument('--use_edge_weight', action='store_true',
    #                     help="whether to consider edge weight in GNN")
    # # Training settings
    # parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimization')
    # parser.add_argument('--epochs', type=int, default=2)
    # parser.add_argument('--num_workers', type=int, default=11)
    # parser.add_argument('--num_negs', type=int, default=1, help='number of negatives for each positive')
    # parser.add_argument('--train_node_embedding', action='store_true',
    #                     help="also train free-parameter node embeddings together with GNN")
    # parser.add_argument('--propagate_embeddings', action='store_true',
    #                     help='propagate the node embeddings using the GCN diffusion operator')
    # parser.add_argument('--loss', default='bce', type=str, help='bce or auc')
    # parser.add_argument('--add_normed_features', dest='add_normed_features', type=str2bool,
    #                     help='Adds a set of features that are normalsied by sqrt(d_i*d_j) to calculate cosine sim')
    # parser.add_argument('--use_RA', type=str2bool, default=False, help='whether to add resource allocation features')
    # # SEAL specific args
    # parser.add_argument('--dynamic_train', action='store_true',
    #                     help="dynamically extract enclosing subgraphs on the fly")
    # parser.add_argument('--dynamic_val', action='store_true')
    # parser.add_argument('--dynamic_test', action='store_true')
    # parser.add_argument('--pretrained_node_embedding', type=str, default=None,
    #                     help="load pretrained node embeddings as additional node features")
    # # Testing settings
    # parser.add_argument('--rep', type=int, default=0, help='the number of repetition of the experiment to run')
    # parser.add_argument('--rep_doc', type=int, default=1, help='debug: the number of repetition of the experiment to doc')


    # parser.add_argument('--use_valedges_as_input', action='store_true')
    # parser.add_argument('--eval_steps', type=int, default=1)
    # parser.add_argument('--log_steps', type=int, default=1)
    # parser.add_argument('--eval_metric', type=str, default='hits',
    #                     choices=('hits', 'mrr', 'auc'))
    # parser.add_argument('--K', type=int, default=100, help='the hit rate @K')
    # # hash settings
    # parser.add_argument('--use_zero_one', type=str2bool, default=0,
    #                     help="whether to use the counts of (0,1) and (1,0) neighbors")
    # parser.add_argument('--floor_sf', type=str2bool, default=0,
    #                     help='the subgraph features represent counts, so should not be negative. If --floor_sf the min is set to 0')
    # parser.add_argument('--hll_p', type=int, default=8, help='the hyperloglog p parameter')
    # parser.add_argument('--minhash_num_perm', type=int, default=128, help='the number of minhash perms')
    # parser.add_argument('--max_hash_hops', type=int, default=2, help='the maximum number of hops to hash')
    # parser.add_argument('--subgraph_feature_batch_size', type=int, default=11000000,
    #                     help='the number of edges to use in each batch when calculating subgraph features. '
    #                          'Reduce or this or increase system RAM if seeing killed messages for large graphs')
    # # wandb settings
    # parser.add_argument('--wandb', action='store_true', help="flag if logging to wandb")
    # parser.add_argument('--wandb_offline', dest='use_wandb_offline',
    #                     action='store_true')  # https://docs.wandb.ai/guides/technical-faq

    # parser.add_argument('--wandb_sweep', action='store_true',
    #                     help="flag if sweeping")  # if not it picks up params in greed_params
    # parser.add_argument('--wandb_watch_grad', action='store_true', help='allows gradient tracking in train function')
    # parser.add_argument('--wandb_track_grad_flow', action='store_true')

    # parser.add_argument('--wandb_entity', default="graph-diffusion-model-link-prediction", type=str)
    # parser.add_argument('--wandb_project', default="link-prediction", type=str)
    # parser.add_argument('--wandb_group', default="testing", type=str, help="testing,tuning,eval")
    # parser.add_argument('--wandb_run_name', default=None, type=str)
    # parser.add_argument('--wandb_output_dir', default='./wandb_output',
    #                     help='folder to output results, images and model checkpoints')
    # parser.add_argument('--wandb_log_freq', type=int, default=1, help='Frequency to log metrics.')
    # parser.add_argument('--wandb_epoch_list', nargs='+', default=[0, 1, 2, 4, 8, 16],
    #                     help='list of epochs to log gradient flow')
    # parser.add_argument('--wandb_notes', nargs='+', help='notes to add to wandb')
    # parser.add_argument('--wandb_tags', type=str, help='tags to add to wandb')
    # parser.add_argument('--log_features', action='store_true', help="log feature importance")
    # parser.add_argument('--save_result', type=bool, default=True, help="save the result to use later for inference")

    parser.add_argument('--config', default="",
                        metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg

    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        config = CN({k: v for k, v in yaml.safe_load(open(args.config, 'r')).items()})

        # load into dict and check 
        if (config.gnn.model.max_hash_hops == 1) and (not config.gnn.model.use_zero_one):
            print("WARNING: (0,1) feature knock out is not supported for 1 hop. Running with all features")
        if config.dataset.name == 'ogbl-ddi':
            config.dataset.use_feature = 0  # dataset has no features
            assert config.gnn.train.sign_k > 0, '--sign_k must be set to > 0 i.e. 1,2 or 3 for ogbl-ddi'

        print(config.gnn.model.use_text, config.dataset.name)

        cfg = merge_from_other_cfg(cfg, config)
    # Update from command line
    cfg.merge_from_list(args.opts)
    # cfg = cfgnode_to_dict(cfg)
    return cfg

cfg_data = set_cfg(CN())

if __name__ == "__main__":
    cfg = update_cfg(cfg_data)
