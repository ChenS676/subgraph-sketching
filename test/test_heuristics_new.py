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

import unittest

import torch
from torch import tensor
import scipy.sparse as ssp
import numpy as np
from src.heuristics_new import HeuristicModels
from src.data_utils.util import get_graph_config

class HeuristicTests(unittest.TestCase):

    def __init__(self,config_dict):

        super(HeuristicTests, self).__init__()

        self.algorithm = config_dict['algorithm']
        self.num_nodes = config_dict['num_nodes']
        self.edge_weight = config_dict['edge_weight']
        self.edge_index = config_dict['edge_index']
        self.pos_test_edges = config_dict['pos_test_edges']
        self.neg_test_edges = config_dict['neg_test_edges']
        self.train_scores_target = config_dict['train_scores_target']
        self.pos_scores_target = config_dict['pos_scores_target']
        self.neg_scores_target = config_dict['neg_scores_target']
        self.iso_test_edges = config_dict['iso_test_edges']
        self.A = ssp.csr_matrix((self.edge_weight, (self.edge_index[:, 0], self.edge_index[:, 1])),
                                shape=(self.num_nodes, self.num_nodes), dtype=float)

    def unit_test(self,test_name, test_truth, edge_index):
        heu = HeuristicModels(self.algorithm, self.A, edge_index)
        heu.test()
        if np.allclose(heu.scores, test_truth):
            print(f'Calcualted {test_name} is the same with ground truth', end='\n\n')

    def regular_test(self):
        print(f'Regular test on algorithm: {self.algorithm}:', end = '\n\n')
        # test train graph
        self.unit_test(test_name='train_scores',
                       test_truth=self.train_scores_target,
                       edge_index=self.edge_index)
        # test positive link
        self.unit_test(test_name='pos_scores',
                       test_truth=self.pos_scores_target,
                       edge_index=self.pos_test_edges)
        # test negative link
        self.unit_test(test_name='neg_scores',
                       test_truth=self.neg_scores_target,
                       edge_index=self.neg_test_edges)

    # def test_RA(self):
    #     train_scores, edge_index = RA(self.A, self.edge_index)
    #     print(train_scores)
    #     self.assertTrue(np.allclose(train_scores, np.array([0, 1 / 2, 0, 0])))
    #     neg_scores, edge_index = RA(self.A, self.neg_test_edges)
    #     self.assertTrue(np.allclose(neg_scores, np.array([1 / 2, 0])))
    #     pos_scores, edge_index = RA(self.A, self.test_edges)
    #     self.assertTrue(np.allclose(pos_scores, np.array([0, 0])))

    def iso_test(self):
        """
        test on isomorphic nodes in the same graph
        """
        print()
        print(f'test iso:')
        
        AA_heu = HeuristicModels('AA', self.A, self.neg_test_edges)
        AA_heu.test()
        print(AA_heu.scores)
        self.assertTrue(AA_heu.scores[0] == AA_heu.scores[1])

        CN_heu = HeuristicModels('CN', self.A, self.neg_test_edges)
        CN_heu.test()
        print(CN_heu.scores)
        self.assertTrue(CN_heu.scores[0] == CN_heu.scores[1])

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
    parser.add_argument('--config_yaml',type=str,default='Graph2_AA.yaml',
                        help='yaml file containing configuration of an algorithm example')
    parser.add_argument('--regular_test',type=int,default=1,
                        help='whether test regular scores or not, including train scores, positive test, negative test')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    config_dict = get_graph_config(args.config_yaml)
    HT = HeuristicTests(config_dict)
    if args.regular_test:
        HT.regular_test()
    if args.config_yaml.split('_')[0].endswith('Iso'):
        HT.iso_test()