"""
run the heuristic baselines resource allocation, common neighbours, personalised pagerank and adamic adar
"""

"""
runner for heuristic link prediction methods. Currently Adamic Adar, Common Neighbours and Personalised PageRank
"""
import time

import argparse
from argparse import Namespace
from ogb.linkproppred import Evaluator
import scipy.sparse as ssp
import torch
import wandb
import numpy as np
import os, sys
sys.path.insert(0, '..')
from src.runners.evaluation import evaluate_auc, evaluate_mrr, evaluate_hits
from src.data_utils.data import get_data

from src.utilities.utils import DEFAULT_DIC, get_pos_neg_edges

"""
A selection of heuristic methods (Personalized PageRank, Adamic Adar and Common Neighbours) for link prediction
"""

import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader

class HeuristicModels():

    def __init__(self,model,A,edge_index):

        super(HeuristicModels, self).__init__()
        self.model = model
        self.A = A
        self.edge_index = edge_index
        
    def test(self):
        if self.model == 'CN':
            self.CN()
        if self.model == 'AA':
            self.AA()
        if self.model == 'rec_AA':
            self.rec_AA()
        if self.model == 'RA':
            self.RA()
        if self.model == 'Jac':
            self.Jac()
        if self.model == 'PA':
            self.PA()


    def CN(self, batch_size=100000):
        """
        Common neighbours
        :param A: scipy sparse adjacency matrix
        :param edge_index: pyg edge_index
        :param batch_size: int
        :return: FloatTensor [edges] of scores, pyg edge_index
        """
        link_loader = DataLoader(range(self.edge_index.size(0)), batch_size)
        scores = []
        for ind in link_loader:
            src, dst = self.edge_index[ind, 0], self.edge_index[ind, 1]
            cur_scores = np.array(np.sum(self.A[src].multiply(self.A[dst]), 1)).flatten()
            scores.append(cur_scores)
        scores = np.concatenate(scores, 0)
        self.scores = torch.FloatTensor(scores)
        print(f'evaluated Common Neighbours for {len(scores)} edges')


    def AA(self, batch_size=100000):
        """
        Adamic Adar
        :param A: scipy sparse adjacency matrix
        :param edge_index: pyg edge_index
        :param batch_size: int
        :return: FloatTensor [edges] of scores, pyg edge_index
        """
        multiplier = 1 / np.log(self.A.sum(axis=0)) # Natural logarithm
        multiplier[np.isinf(multiplier)] = 0 # when N(u)=1, 1/np.log(N(u))=inf
        multiplier[multiplier == -0.0] = 0 # when N(u)=0, 1/np.log(N(u))=-0.0
        # point-wise multiplication of edge weights and edge index 
        A_ = self.A.multiply(multiplier).tocsr()
        # common neighbours 
        link_loader = DataLoader(range(self.edge_index.size(0)), batch_size)
        scores = []
        for ind in link_loader:
            src, dst = self.edge_index[ind, 0], self.edge_index[ind, 1]
            cur_scores = np.array(np.sum(self.A[src].multiply(A_[dst]), 1)).flatten()
            scores.append(cur_scores)
        scores = np.concatenate(scores, 0)
        self.scores = torch.FloatTensor(scores)
        print(f'evaluated Adamic Adar for {len(scores)} edges')

    def rec_AA(self, batch_size=100000):
        """
        which does the same with self.AA()
        """
        # degrees of each node
        multiplier = 1 / np.log(self.A.sum(axis=0))
        multiplier[np.isinf(multiplier)] = 0
        multiplier[multiplier == -0.0] = 0
        # common neighbours 
        link_loader = DataLoader(range(self.edge_index.size(0)), batch_size)
        scores = []
        for ind in link_loader:
            src, dst = self.edge_index[ind, 0], self.edge_index[ind, 1]
            comNeib = self.A[src].multiply(self.A[dst])
            cur_scores = np.array(np.sum(comNeib.multiply(multiplier), 1)).flatten()
            scores.append(cur_scores)
        scores = np.concatenate(scores, 0)
        self.scores = torch.FloatTensor(scores)
        print(f'evaluated Adamic Adar for {len(scores)} edges')


    def RA(self, batch_size=100000):
        """
        Resource Allocation https://arxiv.org/pdf/0901.0553.pdf
        :param A: scipy sparse adjacency matrix
        :param edge_index: pyg edge_index
        :param batch_size: int
        :return: FloatTensor [edges] of scores, pyg edge_index
        """
        multiplier = 1 / self.A.sum(axis=0)
        multiplier[np.isinf(multiplier)] = 0
        A_ = self.A.multiply(multiplier).tocsr()
        link_loader = DataLoader(range(self.edge_index.size(0)), batch_size)
        scores = []
        for ind in link_loader:
            src, dst = self.edge_index[ind, 0], self.edge_index[ind, 1]
            cur_scores = np.array(np.sum(self.A[src].multiply(A_[dst]), 1)).flatten()
            scores.append(cur_scores)
        scores = np.concatenate(scores, 0)
        self.scores = torch.FloatTensor(scores)
        print(f'evaluated Resource Allocation for {len(scores)} edges')


    def Jac(self, batch_size=100000):
        """
        Jaccard: | Neib(x) and Neib(y) | / | Neib(x) or Neib(y) |
        :param A: scipy sparse adjacency matrix
        :param edge_index: pyg edge_index
        :param batch_size: int
        :return: FloatTensor [edges] of scores, pyg edge_index
        """
        A = self.A.toarray()
        link_loader = DataLoader(range(self.edge_index.size(0)), batch_size)
        scores = []
        for ind in link_loader:
            src, dst = self.edge_index[ind, 0], self.edge_index[ind, 1]
            inter = np.sum(np.logical_and(A[src],A[dst]), axis=1) # intersection: Neib(x) and Neib(y)
            union = np.sum(np.logical_or(A[src],A[dst]), axis=1) # union: Neib(x) or Neib(y)
            cur_scores = (inter / union).flatten()
            cur_scores[np.isinf(cur_scores)] = 0
            scores.append(cur_scores)
        scores = np.concatenate(scores, 0)
        self.scores = torch.FloatTensor(scores)
        print(f'evaluated Jaccard for {len(scores)} edges')


    def PA(self, batch_size=100000):
        """
        preferential attachment: | Neib(x) x Neib(y) |
        :param A: scipy sparse adjacency matrix
        :param edge_index: pyg edge_index
        :param batch_size: int
        :return: FloatTensor [edges] of scores, pyg edge_index
        """
        A = self.A.toarray()
        link_loader = DataLoader(range(self.edge_index.size(0)), batch_size)
        scores = []
        for ind in link_loader:
            src, dst = self.edge_index[ind, 0], self.edge_index[ind, 1]
            Neib_src = np.sum(A[src], axis=1) # number of Neib(x)
            Neib_dst = np.sum(A[dst], axis=1) # number of Neib(y)
            cur_scores = Neib_src * Neib_dst
            scores.append(cur_scores)
        scores = np.concatenate(scores, 0)
        self.scores = torch.FloatTensor(scores)
        print(f'evaluated preferential attachment for {len(scores)} edges')


    def PPR(A, edge_index):
        """
        The Personalized PageRank heuristic score.
        Need to install fast_pagerank by "pip install fast-pagerank"
        Too slow for large datasets now.
        :param A: A CSR matrix using the 'message passing' edges
        :param edge_index: The supervision edges to be scored
        :return:
        """
        from fast_pagerank import pagerank_power
        num_nodes = A.shape[0]
        src_index, sort_indices = torch.sort(edge_index[:, 0])
        dst_index = edge_index[sort_indices, 1]
        edge_reindex = torch.stack([src_index, dst_index])
        scores = []
        visited = set([])
        j = 0
        for i in range(edge_reindex.shape[1]):
            if i < j:
                continue
            src = edge_reindex[0, i]
            personalize = np.zeros(num_nodes)
            personalize[src] = 1
            # get the ppr for the current source node
            ppr = pagerank_power(A, p=0.85, personalize=personalize, tol=1e-7)
            j = i 
            # get ppr for all links that start at this source to save recalculating the ppr score
            while edge_reindex[0, j] == src:
                j += 1
                if j == edge_reindex.shape[1]:
                    break
            all_dst = edge_reindex[1, i:j]
            cur_scores = ppr[all_dst]
            if cur_scores.ndim == 0:
                cur_scores = np.expand_dims(cur_scores, 0)
            scores.append(np.array(cur_scores))

        scores = np.concatenate(scores, 0)
        print(f'evaluated PPR for {len(scores)} edges')
        return torch.FloatTensor(scores), edge_reindex



def run(args):
  args = Namespace(**{**DEFAULT_DIC, **vars(args)})
  wandb.init(project=args.wandb_project, config=args, entity=args.wandb_entity)
  # set the correct metric for ogb
  k = 100
  if args.dataset_name == 'ogbl-collab':
    k = 50
  elif args.dataset_name == 'ogbl-ppi':
    k = 20

  for heuristic in [RA, CN, AA, PPR]:
    results_list = []
    for rep in range(args.reps):
      t0 = time.time()
      dataset, splits, directed, eval_metric = get_data(args)
      train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']
      num_nodes = dataset.data.num_nodes
      if 'edge_weight' in train_data:
        train_weight = train_data.edge_weight.view(-1)
        test_weight = test_data.edge_weight.view(-1)
      else:
        train_weight = torch.ones(train_data.edge_index.size(1), dtype=int)
        test_weight = torch.ones(test_data.edge_index.size(1), dtype=int)
      train_edges, val_edges, test_edges = train_data['edge_index'], val_data['edge_index'], test_data['edge_index']
      assert torch.equal(val_edges, train_edges)
      A_train = ssp.csr_matrix((train_weight, (train_edges[0], train_edges[1])),
                               shape=(num_nodes, num_nodes))
      A_test = ssp.csr_matrix((test_weight, (test_edges[0], test_edges[1])),
                              shape=(num_nodes, num_nodes))

      # this function returns transposed edge list of shape [?,2]
      pos_train_edge, neg_train_edge = get_pos_neg_edges(splits['train'])
      pos_val_edge, neg_val_edge = get_pos_neg_edges(splits['valid'])
      pos_test_edge, neg_test_edge = get_pos_neg_edges(splits['test'])

      print(f'results for {heuristic.__name__} (val, test)')
      pos_train_pred, pos_train_edge = heuristic(A_train, pos_train_edge)
      neg_train_pred, neg_train_edge = heuristic(A_train, neg_train_edge)
      pos_val_pred, pos_val_edge = heuristic(A_train, pos_val_edge)
      neg_val_pred, neg_val_edge = heuristic(A_train, neg_val_edge)
      pos_test_pred, pos_test_edge = heuristic(A_test, pos_test_edge)
      neg_test_pred, neg_test_edge = heuristic(A_test, neg_test_edge)

      if args.dataset_name == 'ogbl-citation2':
        evaluator = Evaluator(name=args.dataset_name)
        print(f'evaluating {pos_test_pred.shape} positive samples and {neg_test_pred.shape} negative samples')
        mrr_results = evaluate_mrr(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred,
                                   neg_test_pred)
        print(mrr_results)
        key = 'MRR'
        train_res, val_res, test_res = mrr_results[key]
        res_dic = {f'rep{rep}_Train' + key: 100 * train_res, f'rep{rep}_Val' + key: 100 * val_res,
                   f'rep{rep}_Test' + key: 100 * test_res}
        
        wandb.log(res_dic)
        results_list.append(mrr_results[key])
      else:
        evaluator = Evaluator(name='ogbl-ppa')
        hit_results = evaluate_hits(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred,
                                    pos_test_pred,
                                    neg_test_pred, Ks=[k])
        key = f'Hits@{k}'
        train_res, val_res, test_res = hit_results[key]
        res_dic = {f'rep{rep}_Train' + key: 100 * train_res, f'rep{rep}_Val' + key: 100 * val_res,
                   f'rep{rep}_Test' + key: 100 * test_res}
        wandb.log(res_dic)
        results_list.append(hit_results[key])
        print(hit_results)

      val_pred = torch.cat([pos_val_pred, neg_val_pred])
      val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int),
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
      test_pred = torch.cat([pos_test_pred, neg_test_pred])
      test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int),
                             torch.zeros(neg_test_pred.size(0), dtype=int)])
      auc_results = evaluate_auc(val_pred, val_true, test_pred, test_true)
      print(auc_results)
    if args.reps > 1:
      train_acc_mean, val_acc_mean, test_acc_mean = np.mean(results_list, axis=0) * 100
      test_acc_std = np.sqrt(np.var(results_list, axis=0)[-1]) * 100
      wandb_results = {f'{heuristic.__name__}_test_mean': test_acc_mean, f'{heuristic.__name__}_val_mean': val_acc_mean,
                       f'{heuristic.__name__}_train_mean': train_acc_mean,
                       f'{heuristic.__name__}_test_acc_std': test_acc_std}
      wandb.log(wandb_results)
      print(wandb_results)
    print(f'{heuristic.__name__} ran in {time.time() - t0:.1f} s for {args.reps} reps')
  wandb.finish()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_name', type=str, default='Cora',
                      choices=['Cora', 'producer', 'Citeseer', 'Pubmed', 'ogbl-ppa', 'ogbl-collab', 'ogbl-ddi',
                               'ogbl-citation2'])
  parser.add_argument('--wandb_entity', default="link-prediction", type=str)
  parser.add_argument('--wandb_project', default="link-prediction", type=str)
  parser.add_argument('--reps', type=int, default=1, help='the number of repetition of the experiment to run')
  parser.add_argument('--sample_size', type=int, default=None,
                      help='the number of training edges to sample. Currently only implemented for producer data')

  args = parser.parse_args()
  print(args)
  run(args)
