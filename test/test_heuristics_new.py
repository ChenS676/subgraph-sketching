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
        print(f'scores: {heu.scores}')
        print(f'ground truth: {test_truth}')
        if np.allclose(heu.scores, test_truth):
            print(f'Calcualted {test_name} is the same with ground truth', end='\n\n')
        else:
            print(f'Failed, calculated values != ground truth', end='\n\n')

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

    def iso_test(self):
        """
        test on isomorphic nodes in the same graph
        """
        if self.iso_test_edges is None:
            print('There are no isomorphic nodes in current graph')
        else:
            print(f'Test isomorphic nodes with algorithm {self.algorithm}:')
            iso_heu = HeuristicModels(self.algorithm, self.A, self.iso_test_edges)
            iso_heu.test()
            print(f'node: {self.iso_test_edges[0]}, values {iso_heu.scores[0]}')
            print(f'node: {self.iso_test_edges[1]}, values {iso_heu.scores[1]}')
            print('Vales are {}'.format('same' if iso_heu.scores[0] == iso_heu.scores[1] else 'not same'))

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
    parser.add_argument('--config_yaml',type=str,default='GraphIso_RA.yaml',
                        help='yaml file containing configuration of an algorithm example')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    config_dict = get_graph_config(args.config_yaml)
    HT = HeuristicTests(config_dict)
    HT.regular_test()
    HT.iso_test()
