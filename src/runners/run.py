"""
main module
"""
import argparse
import time
import warnings

import sys, os
import random
import numpy as np
import torch
from ogb.linkproppred import Evaluator
sys.path.insert(0, '..')
torch.set_printoptions(precision=4)
import wandb
# when generating subgraphs the supervision edge is deleted, which triggers a SparseEfficiencyWarning, but this is
# not a performance bottleneck, so suppress for now
from scipy.sparse import SparseEfficiencyWarning
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

from src.data import get_data, get_loaders
from src.models.elph import ELPH, BUDDY
from src.models.seal import SEALDGCNN, SEALGCN, SEALGIN, SEALSAGE
from src.utils import ROOT_DIR, print_model_params, select_embedding, str2bool, save_metrics_to_csv
from src.wandb_setup import initialise_wandb
from src.runners.train import get_train_func
from src.runners.inference import test
from pdb import set_trace as bp 

from src.config_load import cfg_data as cfg
from src.config_load import update_cfg, cfgnode_to_dict, recursive_search_key

def print_results_list(results_list):
    for idx, res in enumerate(results_list):
        print(f'repetition {idx}: test {res[0]:.2f}, val {res[1]:.2f}, train {res[2]:.2f}')

def set_seed(seed):
    """
    setting a random seed for reproducibility and in accordance with OGB rules
    @param seed: an integer seed
    @return: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run(args):
    initialise_wandb(args)
    # combine cfg and args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"executing on {device}")
    results_list = []
    train_func = get_train_func(args)

    # for rep in range(args.reps):
    set_seed(0) # set_seed(rep)
    dataset, splits, directed, eval_metric = get_data(args)
    train_loader, train_eval_loader, val_loader, test_loader = get_loaders(args, dataset, splits, directed)
    if args.dataset.name.startswith('ogbl'):  # then this is one of the ogb link prediction datasets
        evaluator = Evaluator(name=args.dataset_name)
    else:
        evaluator = Evaluator(name='ogbl-ppa')  # this sets HR@100 as the metric
    
    emb = select_embedding(args, dataset.data.num_nodes, device)
    model, optimizer = select_model(args, dataset, emb, device)

    val_res = test_res = best_epoch = 0
    rep = 0
    print_model_params(model)
    for epoch in range(args.epochs):
        t0 = time.time()
        loss = train_func(model, optimizer, train_loader, args, device)
        if (epoch + 1) % args.eval_steps == 0:
            results = test(model, evaluator, train_eval_loader, val_loader, test_loader, args, device,
                            eval_metric=eval_metric)
            for key, result in results.items():
                train_res, tmp_val_res, tmp_test_res = result
                if tmp_val_res > val_res:
                    val_res = tmp_val_res
                    test_res = tmp_test_res
                    best_epoch = epoch
                res_dic = {f'rep{rep}_loss': loss, f'rep{rep}_Train' + key: 100 * train_res,
                            f'rep{rep}_Val' + key: 100 * val_res, f'rep{rep}_tmp_val' + key: 100 * tmp_val_res,
                            f'rep{rep}_tmp_test' + key: 100 * tmp_test_res,
                            f'rep{rep}_Test' + key: 100 * test_res, f'rep{rep}_best_epoch': best_epoch,
                            f'rep{rep}_epoch_time': time.time() - t0, 'epoch_step': epoch}
                if args.wandb:
                    wandb.log(res_dic)
                to_print = f'Epoch: {epoch:02d}, Best epoch: {best_epoch}, Loss: {loss:.4f}, Train: {100 * train_res:.2f}%, Valid: ' \
                            f'{100 * val_res:.2f}%, Test: {100 * test_res:.2f}%, epoch time: {time.time() - t0:.1f}'
                print(key)
                print(to_print)
        # TODO change when writing
        # if args.reps == args.rep_doc - 1 :
            results_list.append([test_res, val_res, train_res])
            print_results_list(results_list)
    # if args.reps > 1:
        test_acc_mean, val_acc_mean, train_acc_mean = np.mean(results_list, axis=0) * 100
        test_acc_std = np.sqrt(np.var(results_list, axis=0)[0]) * 100
        val_acc_std = np.sqrt(np.var(results_list, axis=0)[1]) * 100

        id = f'{args.dataset.name}-{args.model}-{args.use_text}'
        wandb_results = {'id': id, 
                         'test_mean': test_acc_mean, 'val_mean': val_acc_mean, 'train_mean': train_acc_mean,
                            'test_acc_std': test_acc_std, 'val_acc_std': val_acc_std, 'epoch': args.epochs}
        del id
        print(wandb_results)
    if args.wandb:
        wandb.log(wandb_results)
    if args.wandb:
        wandb.finish()
    if args.save_model:
        path = f'{ROOT_DIR}/saved_models/{args.dataset.name}'
        torch.save(model.state_dict(), path)
    if args.save_result:
        # ToDo save excel metrics. 
        save_metrics_to_csv(wandb_results)
        print('saved.')
from copy import deepcopy
def select_model(parser, dataset, emb, device):
    args = deepcopy(parser)
    args.model = args.gnn.model.name
    if args.model == 'SEALDGCNN':
        model = SEALDGCNN(args.train.hidden_channels, args.num_seal_layers, args.gnn.model.max_z, 
                          args.gnn.model.sortpool_k,
                          dataset, args.gnn.model.dynamic_train, use_feature=args.data.dataloader.use_feature,
                          node_embedding=emb).to(device)
    elif args.model == 'SEALSAGE':
        model = SEALSAGE(args.hidden_channels, args.num_seal_layers, args.max_z, dataset.dataloader.num_features,
                         args.use_feature, node_embedding=emb, dropout=args.dropout).to(device)
    elif args.model == 'SEALGCN':
        model = SEALGCN(args.hidden_channels, args.num_seal_layers, args.max_z, dataset.num_features,
                        args.use_feature, node_embedding=emb, dropout=args.dropout, pooling=args.seal_pooling).to(
            device)
    elif args.model == 'SEALGIN':
        model = SEALGIN(args.hidden_channels, args.num_seal_layers, args.max_z, dataset.num_features,
                        args.use_feature, node_embedding=emb, dropout=args.dropout).to(device)
    elif args.model == 'BUDDY':
        model = BUDDY(args, dataset.num_features, node_embedding=emb).to(device)
    elif args.model == 'ELPH':
        model = ELPH(args, dataset.num_features, node_embedding=emb).to(device)
    else:
        raise NotImplementedError
    parameters = list(model.parameters())
    if args.train.node_embedding:
        torch.nn.init.xavier_uniform_(emb.weight)
        parameters += list(emb.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for param in parameters for p in param)
    print(f'Total number of parameters is {total_params}')
    if args.model == 'DGCNN':
        print(f'SortPooling k is set to {model.k}')
    return model, optimizer


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    run(cfg)