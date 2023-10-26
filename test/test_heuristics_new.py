"""
testing the simple heuristics Personalized PageRank, Adamic Adar and Common Neighbours

This is the directed test graph
   -> 0 -> 1 <-
   |          |
    --- 2 <---
"""

from inspect import getsourcefile
import os
import sys
import argparse
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

import math
import unittest

import torch
from torch import tensor
import scipy.sparse as ssp
import numpy as np
# from src.heuristics_new import rec_AA, AA, PPR, CN, RA
from src.heuristics_new import HeuristicModels
from src.data_utils.util import get_graph_config

class HeuristicTests(unittest.TestCase):

    def __init__(self,config_dict):

        super(HeuristicTests, self).__init__()

        self.algorithm = config_dict['algorithm']
        self.num_nodes = config_dict['num_nodes']
        self.edge_weight = config_dict['edge_weight']
        self.edge_index = config_dict['edge_index']

        print(f'Test on algorithm: {self.algorithm}:')

        self.setUp()
        # test train graph
        self.test(test_name='train_scores',
                  test_truth=config_dict['train_scores_target'],
                  edge_index=config_dict['edge_index'])
        # test positive link
        self.test(test_name='pos_scores',
                  test_truth=config_dict['pos_scores_target'],
                  edge_index=config_dict['pos_test_edges'])
        # test negative link
        self.test(test_name='neg_scores',
                  test_truth=config_dict['neg_scores_target'],
                  edge_index=config_dict['neg_test_edges'])

        
    def setUp(self):
        
        self.A = ssp.csr_matrix((self.edge_weight, (self.edge_index[:, 0], self.edge_index[:, 1])),
                                shape=(self.num_nodes, self.num_nodes), dtype=float)
        # create a graph with 2 isomorphic nodes 2 & 3
        self.iso_edge_index = tensor([[2, 2, 3, 3, 4, 0], [1, 4, 1, 4, 0, 1]]).t()
        self.iso_edge_weight = torch.ones(self.iso_edge_index.size(0), dtype=int)
        self.iso_test_edges = tensor([[2, 3], [0, 0]]).t()
        self.iso_num_nodes = 5

    def AA2_setUp(self):
        self.A = np.array([[0, 1 ,0 ,0 , 1], [1, 0 , 1 , 1 ,1], [0 , 1, 0 , 1 , 1], [0, 1, 1 , 0, 1], [1, 1 , 1 , 1 , 0 ]])
        row, col = self.A.nonzero()
        self.num_nodes = self.A.shape[0]
        self.edge_index =  tensor([np.array(row), np.array(col)]).t()
        self.edge_weight = torch.ones(self.edge_index.size(0), dtype=torch.int)
        self.A = ssp.csr_matrix((self.edge_weight, (self.edge_index[:, 0], self.edge_index[:, 1])),
                                shape=(self.num_nodes, self.num_nodes), dtype=float)

    def test(self,test_name, test_truth, edge_index):
        heu = HeuristicModels(self.algorithm, self.A, edge_index)
        heu.test()
        print(f'{test_name}: {heu.scores}')
        self.assertTrue(np.allclose(heu.scores, test_truth))

    # def test_RA(self):
    #     train_scores, edge_index = RA(self.A, self.edge_index)
    #     print(train_scores)
    #     self.assertTrue(np.allclose(train_scores, np.array([0, 1 / 2, 0, 0])))
    #     neg_scores, edge_index = RA(self.A, self.neg_test_edges)
    #     self.assertTrue(np.allclose(neg_scores, np.array([1 / 2, 0])))
    #     pos_scores, edge_index = RA(self.A, self.test_edges)
    #     self.assertTrue(np.allclose(pos_scores, np.array([0, 0])))

    # def test_iso_graph(self):
    #     A = ssp.csr_matrix((self.iso_edge_weight, (self.iso_edge_index[:, 0], self.iso_edge_index[:, 1])),
    #                        shape=(self.iso_num_nodes, self.iso_num_nodes))
    #     aa_test_scores, edge_index = AA(A, self.iso_test_edges)
    #     print(aa_test_scores)
    #     self.assertTrue(aa_test_scores[0] == aa_test_scores[1])
    #     cn_test_scores, edge_index = CN(A, self.iso_test_edges)
    #     print(cn_test_scores)
    #     self.assertTrue(cn_test_scores[0] == cn_test_scores[1])
    #     ppr_test_scores, edge_index = PPR(A, self.iso_test_edges)
    #     print(ppr_test_scores)
    #     self.assertTrue(ppr_test_scores[0] == ppr_test_scores[1])

    # def test_PPR(self):
    #     train_scores, edge_index = PPR(self.A, self.edge_index)
    #     print(train_scores)
    #     self.assertTrue(np.allclose(train_scores, np.array([0, 0.5, 0, 0])))
    #     neg_scores, edge_index = PPR(self.A, self.neg_test_edges)
    #     self.assertTrue(np.allclose(neg_scores, np.array([0.5, 0])))
    #     pos_scores, edge_index = PPR(self.A, self.test_edges)
    #     self.assertTrue(np.allclose(pos_scores, np.array([0, 0])))

def get_args():

    parser = argparse.ArgumentParser(description='Heuristic graph algorithms')
    # add arguments
    parser.add_argument('--algorithm',type=str,default='AA',
                        choices=['AA','rec_AA','CN','RA','iso_graph','PPR'],
                        help='select an algorithm')
    parser.add_argument('--num_nodes',type=int,default=5,help='numbe of nodes in graph')

    parser.add_argument('--edge_index',
                        type=torch.tensor,
                        default=tensor([[0,0,0,1,1,3,3], [1,2,3,2,3,0,1]]).t(),
                        choices=[
                            tensor([[0,2,2,1], [1,0,1,2]]).t(), # default example
                            tensor([[0,0,0,1,1,3,3], [1,2,3,2,3,0,1]]).t(), # e.g.1
                            tensor([[0,0,1,3], [2,3,2,1]]).t(), # e.g.2
                            tensor([[0,0,0,1,2,3,4], [2,3,4,2,4,1,1]]).t(), # e.g.3
                        ],
                        help='structure of graph')
    
    parser.add_argument('--pos_test_edges',
                        type=torch.tensor,
                        default=tensor([[0, 1], [1, 2]]).t(),
                        choices=[
                            tensor([[0, 1], [1, 2]]).t(), # default example
                        ],
                        help='positive test edges')
    
    parser.add_argument('--neg_test_edges',
                        type=torch.tensor,
                        default=tensor([[0, 1], [2, 0]]).t(),
                        choices=[
                            tensor([[0, 1], [2, 0]]).t(), # default example
                        ],
                        help='negaive test edges')
    
    parser.add_argument('--train_scores_target',
                        type=np.array,
                        default=np.array([2/math.log(2), 0, 1/math.log(2), 0, 0, 1/math.log(2), 0]),
                        choices=[
                            np.array([2/math.log(2), 0, 1/math.log(2), 0, 0, 1/math.log(2), 0]), # e.g.1 AA
                            np.array([2, 0, 1, 0, 0, 1, 0]), # e.g.1 CN
                        ],
                        help='true score of edge_index, please manually calculate')
    
    parser.add_argument('--neg_scores_target',
                        type=np.array,
                        default=np.array([0, 2/math.log(2)]),
                        choices=[
                            np.array([0, 2/math.log(2)]), # e.g.1 AA
                            np.array([0, 2]), # e.g.1 CN
                        ],
                        help='true score of neg_test_edges, please manually calculate')
    
    parser.add_argument('--pos_scores_target',
                        type=np.array,
                        default=np.array([2/math.log(2), 0]),
                        choices=[
                            np.array([2/math.log(2), 0]), # e.g.1 AA
                            np.array([2, 0]), # e.g.1 CN
                        ],
                        help='true score of pos_test_edges, please manually calculate')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # general config
    args = get_args()
    config_dict = {'algorithm':args.algorithm,
                   'num_nodes': args.num_nodes,
                   'edge_weight': torch.ones(args.edge_index.size(0), dtype=torch.int),
                   'edge_index': args.edge_index,
                   'pos_test_edges': args.pos_test_edges,
                   'neg_test_edges': args.neg_test_edges,
                   }

    # default example
    if torch.equal(args.edge_index, tensor([[0,2,2,1], [1,0,1,2]]).t()):
        # AA
        if args.algorithm == 'AA':
            config_dict['train_scores_target'] = np.array([0, 1 / math.log(2), 0, 0])
            config_dict['neg_scores_target'] = np.array([1 / math.log(2), 0])
            config_dict['pos_scores_target'] = np.array([0, 0])
        
        # rec_AA
        if args.algorithm == 'rec_AA':
            config_dict['train_scores_target'] = np.array([0, 1 / math.log(2), 0, 0])
            config_dict['neg_scores_target'] = np.array([1 / math.log(2), 0])
            config_dict['pos_scores_target'] = np.array([0, 0])

        # CN
        if args.algorithm == 'CN':
            config_dict['train_scores_target'] = np.array([0, 1, 0, 0])
            config_dict['neg_scores_target'] = np.array([1, 0])
            config_dict['pos_scores_target'] = np.array([0, 0])
    # customized example
    else:
        config_dict['train_scores_target'] = args.train_scores_target
        config_dict['neg_scores_target'] = args.neg_scores_target
        config_dict['pos_scores_target'] = args.pos_scores_target

    test = HeuristicTests(config_dict)