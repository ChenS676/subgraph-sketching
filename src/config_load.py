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

def flatten_cfg_node(cfg_node, parent_key='', sep='_'):
    flat_dict = {}
    for k, v in cfg_node.items():
        new_key = f"{k}" if parent_key else k
        if isinstance(v, CN):
            flat_dict.update(flatten_cfg_node(v, new_key, sep=sep))
        else:
            flat_dict[new_key] = v
    return flat_dict


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

def merge_dicts(dict1, dict2, default_value=''):
    result = {}
    
    for key in set(dict1.keys()) | set(dict2.keys()):
        value1 = dict1.get(key, default_value)
        value2 = dict2.get(key, default_value)

        if isinstance(value1, dict) and isinstance(value2, dict):
            result[key] = merge_dicts(value1, value2, default_value=default_value)
        else:
            result[key] = max_info(value1, value2)

    return result

def max_info(val1, val2):
    if val1 == val2:
        return val1 
    elif val1 == val2 == '':
        return ''
    elif val1 == None and val2 == None:  
        return None
    elif val1 == '' and val2 != '': 
        return val2 
    elif val1 != '' and val2 == '': 
        return val1 
    else: 
        print((f'val1: {val1}, val2: {val2}'))
        raise ValueError(f'val1: {val1}, val2: {val2}')
    

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

from typing import Dict

def count_elements(d):
    count = 0
    for value in d.values():
        if isinstance(value, dict):
            count += count_elements(value)
        else:
            count += 1
    return count

def merge_from_other_cfg(cfg, config):
    cfg_dict = cfgnode_to_dict(cfg)
    config_dict = cfgnode_to_dict(config)
    
    merged_dict1 = merge_dicts(config_dict, cfg_dict)

    merged_from_other_cfg = CN(merged_dict1)
    return merged_from_other_cfg




def set_cfg(cfg):
    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Cuda device number, used for machine with multiple gpus
    cfg.device = CN()
    cfg.device.index = 0
    cfg.device.num_workers = 11
    # Whether fix the running seed to remove randomness

    cfg.seed = 1
    # Number of runs with random init
    cfg.runs = 4
    cfg.log = CN()
    cfg.log.log_features = False
    cfg.log.log_steps = 1
    cfg.log.loss = 'bce'
    cfg.log.lr = 0.0001

    cfg.gnn = CN()
    # ------------------------------------------------------------------------ #
    # GNN Model options
    # ------------------------------------------------------------------------ #
    cfg.gnn.model = CN()
    # GNN model name
    cfg.gnn.model.mname = 'ELPH'
    # Number of gnn layers
    cfg.gnn.model.K = 100
    cfg.gnn.model.add_normed_features = False
    cfg.gnn.model.batch_size = 1024
    cfg.gnn.model.cache_subgraph_features = False
    cfg.gnn.model.config = ''
    cfg.gnn.model.max_dist = 4
    cfg.gnn.model.max_hash_hops = 2
    cfg.gnn.model.max_nodes_per_hop = ''
    cfg.gnn.model.max_z = 1000
    cfg.gnn.model.minhash_num_perm = 128
    cfg.gnn.model.node_label = 'drnl'
    cfg.gnn.model.num_hops = 1
    cfg.gnn.model.num_negs = 1
    cfg.gnn.model.num_seal_layers = 3
    cfg.gnn.model.opts = []
    cfg.gnn.model.preprocessing = ''
    cfg.gnn.model.pretrained_node_embedding = ''
    cfg.gnn.model.propagate_embeddings = False
    cfg.gnn.model.ratio_per_hop = 1.0
    cfg.gnn.model.rep = 0
    cfg.gnn.model.rep_doc = 1
    cfg.gnn.model.use_RA = False
    cfg.gnn.model.use_edge_weight = False
    cfg.gnn.model.use_struct_feature = True
    cfg.gnn.model.use_text = True
    cfg.gnn.model.use_valedges_as_input = False
    cfg.gnn.model.use_wandb_offline = False

    # ------------------------------------------------------------------------ #
    # GNN Training options
    # ------------------------------------------------------------------------ #
    cfg.gnn.train = CN()
    # Use PyG or DGL
    cfg.gnn.train.use_dgl = False
    # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
    cfg.gnn.train.weight_decay = 0.0

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

    cfg.gnn.train.dropout = 0.5
    cfg.gnn.train.dynamic_test = False
    cfg.gnn.train.dynamic_train = False
    cfg.gnn.train.dynamic_val = False
    cfg.gnn.train.mepochs = 100
    cfg.gnn.train.eval_batch_size = 1000000
    cfg.gnn.train.eval_metric = 'hits'
    cfg.gnn.train.eval_steps = 1
    cfg.gnn.train.feature_dropout = 0.5
    cfg.gnn.train.feature_prop = 'gcn'
    cfg.gnn.train.floor_sf = 0
    cfg.gnn.train.hidden_channels = 1024
    cfg.gnn.train.hll_p = 8
    cfg.gnn.train.label_dropout = 0.5
    cfg.gnn.train.label_pooling = 'add'
    cfg.gnn.train.load_features = False
    cfg.gnn.train.load_hashes = False
    cfg.gnn.train.save_model = False
    cfg.gnn.train.save_result = True
    cfg.gnn.train.seal_pooling = 'edge'
    cfg.gnn.train.sign_dropout = 0.5
    cfg.gnn.train.sign_k = 0
    cfg.gnn.train.sortpool_k = 0.6
    cfg.gnn.train.subgraph_feature_batch_size = 11000000
    cfg.gnn.train.use_zero_one = 0
    cfg.gnn.train.test_samples = inf
    cfg.gnn.train.train_cache_size = inf
    cfg.gnn.train.node_embedding = False
    cfg.gnn.train.train_samples = inf
    cfg.gnn.train.val_samples = inf
    # ------------------------------------------------------------------------ #
    # LM Model options
    # ------------------------------------------------------------------------ #
    cfg.lm = CN()


    cfg.lm.model = CN()
    # LM model name
    cfg.lm.model.name = 'microsoft/deberta-base'
    cfg.lm.model.feat_shrink = ""

    # ------------------------------------------------------------------------ #
    # LM Training options
    # ------------------------------------------------------------------------ #
    cfg.lm.train = CN()
    #  Number of samples computed once per batch per device
    cfg.lm.train.lbatch_size = 9
    # Number of training steps for which the gradients should be accumulated
    cfg.lm.train.grad_acc_steps = 1
    # Base learning rate
    cfg.lm.train.lr = 2e-5
    # Maximal number of epochs
    cfg.lm.train.lepochs = 0.05
    # The number of warmup steps
    cfg.lm.train.lwarmup_epochs = 0.6
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

    cfg.dataset = CN()
    # root of dataset 
    cfg.dataset.name = 'cora'
    cfg.dataset.feature_type = 'TA'
    cfg.dataset.use_feature = False
    cfg.dataset.use_text = False
    cfg.dataset.val_pct = 0.1
    cfg.dataset.test_pct = 0.2

    cfg.wandb = CN()
    cfg.wandb.wenable = True
    cfg.wandb.wandb_entity = 'graph-diffusion-model-link-prediction'
    cfg.wandb.wandb_epoch_list = [
        - 0
        - 1
        - 2
        - 4
        - 8
        - 16
    ]
    cfg.wandb.use_wandb_offline = False
    cfg.wandb.wandb_group = 'reconstruct'
    cfg.wandb.wandb_log_freq = 1
    cfg.wandb.wandb_notes = None
    cfg.wandb.wandb_output_dir = './wandb_output'
    cfg.wandb.wandb_project = 'link-prediction'
    cfg.wandb.wandb_run_name = ''
    cfg.wandb.wandb_sweep = False
    cfg.wandb.wandb_tags = 'baseline'
    cfg.wandb.wandb_track_grad_flow = False
    cfg.wandb.wandb_watch_grad = False
    cfg.wandb.weight_decay = 0
    cfg.wandb.use_wandb_offline = False
    cfg.wandb.year = 0
    cfg.wandb.wandb_track_grad_flow = False

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


def flatten_dict(d, parent_key='', sep='_'):
    flat_dict = {}
    for k, v in d.items():
        new_key = f"{k}" if parent_key else k
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(v, new_key, sep=sep))
        else:
            flat_dict[new_key] = v
    return flat_dict


def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser(description='Efficient Link Prediction with Hashes (ELPH)')
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
    cfg = CN(flatten_cfg_node(cfg))
    # load into dict and check 
    if (cfg.max_hash_hops == 1) and (not cfg.use_zero_one):
        print("WARNING: (0,1) feature knock out is not supported for 1 hop. Running with all features")
    if cfg.name == 'ogbl-ddi':
        cfg.use_feature = 0  # dataset has no features
        assert cfg.sign_k > 0, '--sign_k must be set to > 0 i.e. 1,2 or 3 for ogbl-ddi'

    print(cfg.use_text, cfg.name)

        # cfg = merge_from_other_cfg(cfg, config)
    # Update from command line
    cfg.merge_from_list(args.opts)
    # cfg = cfgnode_to_dict(cfg)
    return cfg

cfg_data = set_cfg(CN())

if __name__ == "__main__":
    cfg = update_cfg(cfg_data)
