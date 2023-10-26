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


    def RA(A, edge_index, batch_size=100000):
        """
        Resource Allocation https://arxiv.org/pdf/0901.0553.pdf
        :param A: scipy sparse adjacency matrix
        :param edge_index: pyg edge_index
        :param batch_size: int
        :return: FloatTensor [edges] of scores, pyg edge_index
        """
        multiplier = 1 / A.sum(axis=0)
        multiplier[np.isinf(multiplier)] = 0
        A_ = A.multiply(multiplier).tocsr()
        link_loader = DataLoader(range(edge_index.size(0)), batch_size)
        scores = []
        for ind in tqdm(link_loader):
            src, dst = edge_index[ind, 0], edge_index[ind, 1]
            cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
            scores.append(cur_scores)
        scores = np.concatenate(scores, 0)
        print(f'evaluated Resource Allocation for {len(scores)} edges')
        return torch.FloatTensor(scores), edge_index


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

