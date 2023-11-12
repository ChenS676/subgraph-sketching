"""
Read and split ogb and planetoid datasets
"""

import os
import time

import torch
from torch.utils.data import DataLoader
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   to_undirected)

from torch_geometric.loader import DataLoader as pygDataLoader
import wandb

from src.utils import ROOT_DIR, get_same_source_negs
from src.lcc import get_largest_connected_component, remap_edges, get_node_mapper
from src.datasets.seal import get_train_val_test_datasets
from src.datasets.elph import get_hashed_train_val_test_datasets, make_train_eval_data
import json, sys
from pdb import set_trace as bp
from yacs.config import CfgNode
from .config_load import cfg_data as cfg
import os.path as osp
from typing import Any, Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import InMemoryDataset, download_url, Dataset
from torch_geometric.io import read_planetoid_data
from IPython import embed

dataset_path = {
    'cora':
            { 
            'root': '',
            'original': 'dataset/cora_orig/cora',
            'papers': 'dataset/cora_orig/mccallum/cora/papers',
            'extractions': 'dataset/cora_andrew_mccallum/extractions/'
			},
    'pubmed': 
            {
            'root': '',
            'original': 'dataset/PubMed_orig/data/',
            'abs_ti': 'dataset/PubMed_orig/pubmed.json'
            },
    'arxiv':
        {
            'root': '',
            'original': 'dataset/ogbn_arxiv_orig/',
            'abs_ti': 'dataset/ogbn_arxiv_orig/titleabs.tsv'
        }

    }


def get_loaders(args, dataset, splits, directed):
    train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']
    if args.gnn.model.name in {'ELPH', 'BUDDY'}:
        train_dataset, val_dataset, test_dataset = get_hashed_train_val_test_datasets(dataset, train_data, val_data,
                                                                                      test_data, args, directed)
    else:
        t0 = time.time()
        train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(dataset, train_data, val_data, test_data,
                                                                               args)
        print(f'SEAL preprocessing ran in {time.time() - t0} s')
        if args.wandb:
            wandb.log({"seal_preprocessing_time": time.time() - t0})

    dl = DataLoader if args.gnn.model.name in {'ELPH', 'BUDDY'} else pygDataLoader
    train_loader = dl(train_dataset, batch_size=args.dataset.dataloader.batch_size,
                      shuffle=True, num_workers=args.device.num_workers)
    # as the val and test edges are often sampled they also need to be shuffled
    # the citation2 dataset has specific negatives for each positive and so can't be shuffled
    shuffle_val = False if args.dataset.name.startswith('ogbl-citation') else True
    val_loader = dl(val_dataset, batch_size=args.dataset.dataloader.batch_size, 
                    shuffle=shuffle_val,
                    num_workers=args.device.num_workers)
    shuffle_test = False if args.dataset.name.startswith('ogbl-citation') else True
    test_loader = dl(test_dataset, batch_size=args.dataset.dataloader.batch_size, shuffle=shuffle_test,
                     num_workers=args.device.num_workers)
    if (args.dataset.name == 'ogbl-citation2') and (args.gnn.model.name in {'ELPH', 'BUDDY'}):
        train_eval_loader = dl(
            make_train_eval_data(args, train_dataset, train_data.num_nodes,
                                  n_pos_samples=5000), batch_size=args.dataset.dataloader.batch_size, shuffle=False,
            num_workers=args.device.num_workers)
    else:
        # todo change this so that eval doesn't have to use the full training set
        train_eval_loader = train_loader

    return train_loader, train_eval_loader, val_loader, test_loader


def get_data(args):
    """
    Read the dataset and generate train, val and test splits.
    For GNN link prediction edges play 2 roles 1/ message passing edges 2/ supervision edges
    - train message passing edges = train supervision edges
    - val message passing edges = train supervision edges
    val supervision edges are disjoint from the training edges
    - test message passing edges = val supervision + train message passing (= val message passing)
    test supervision edges are disjoint from both val and train supervision edges
    :param args: arguments Namespace object
    :return: dataset, dic splits, bool directed, str eval_metric
    # TODO add embedding for cora, arxiv, and pubmed for NLP method 
    """
    include_negatives = True
    dataset_name = args.dataset.name
    use_text = args.dataset.use_text
    val_pct = args.dataset.val_pct
    test_pct = args.dataset.test_pct
    use_lcc_flag = True
    directed = False
    eval_metric = 'hits'
    splits = None

    # config all
    # https://github.com/melifluos/subgraph-sketching/blob/main/src/data.py#L82
    path = os.path.join(ROOT_DIR, 'dataset', dataset_name)
    print(f'reading data from: {path}')
    if dataset_name.startswith('ogbl'):
        # for collab, ppa, ddi, citation2
        use_lcc_flag = False
        dataset = PygLinkPropPredDataset(name=dataset_name, root=path)
        if dataset_name == 'ogbl-ddi':
            dataset.data.x = torch.ones((dataset.data.num_nodes, 1))
            dataset.data.edge_weight = torch.ones(dataset.data.edge_index.size(1), dtype=int)

    # for ogbn-arxiv and use text false
    elif dataset_name.startswith('ogbn') and not use_text:
        use_lcc_flag = False
        dataset = PygNodePropPredDataset(name=dataset_name, root=path)

        # dataset.data.x = torch.ones((dataset.data.num_nodes, 1))
        # dataset.data.edge_weight = torch.ones(dataset.data.edge_index.size(1), dtype=int)        
    elif dataset_name in ['cora', 'pubmed', 'ogbn-arxiv'] and use_text:# 'ogbn-products', 'tape-arxiv23'], "Invalid dataset name"
        dataset = Textgraph(cfg, dataset_name, use_text)
        directed = False 
        splits = dataset.splits
        if dataset_name.startswith('ogbn'):
            use_lcc_flag = False
    else:
        # with default use text wrong
        dataset = Planetoid(path, dataset_name)

    # set the metric
    if dataset_name.startswith('ogbl-citation'):
        eval_metric = 'mrr'
        directed = True
    print(f"number of nodes: {dataset.data.num_nodes}")
    if use_lcc_flag:
        dataset = use_lcc(dataset)

    undirected = not directed

    if dataset_name.startswith('ogbl'):  # use the built in splits
        data = dataset[0]
        split_edge = dataset.get_edge_split()
        #############################################
        if dataset_name == 'ogbl-collab' and args.year > 0:  # filter out training edges before args.year
            data, split_edge = filter_by_year(data, split_edge, args.year)
        splits = get_ogb_data(data, split_edge, dataset_name, args.num_negs)
        #############################################
    elif dataset_name.startswith('ogbn'):  # use the built in splits
        
        transform = RandomLinkSplit(is_undirected=True, num_val=0.1, num_test=0.2,
                                    add_negative_train_samples=True)
        train_data, val_data, test_data = transform(dataset.data)
        # TODO implement custom edge split read the paper and check the code
        splits = {'train': train_data, 'valid': val_data, 'test': test_data}
    else:  # use the random splits
        transform = RandomLinkSplit(is_undirected=undirected, num_val=val_pct, num_test=test_pct,
                                    add_negative_train_samples=include_negatives)
        print(f'using random splits with val_pct={val_pct} and test_pct={test_pct}')
        print(f'{dataset.data.x.shape[0]}')
        train_data, val_data, test_data = transform(dataset.data)
        splits = {'train': train_data, 'valid': val_data, 'test': test_data}
        for v in splits.values():
            print(np.sum(np.array(v.train_mask)))
    return dataset, splits, directed, eval_metric


def filter_by_year(data, split_edge, year):
    """
    remove edges before year from data and split edge
    @param data: pyg Data, pyg SplitEdge
    @param split_edges:
    @param year: int first year to use
    @return: pyg Data, pyg SplitEdge
    """
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= year).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()
    # create adjacency matrix
    new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    return data, split_edge


def get_ogb_data(data, split_edge, dataset_name, num_negs=1):
    """
    ogb datasets come with fixed train-val-test splits and a fixed set of negatives against which to evaluate the test set
    The dataset.data object contains all of the nodes, but only the training edges
    @param dataset:
    @param use_valedges_as_input:
    @return:
    """
    if num_negs == 1:
        negs_name = f'{ROOT_DIR}/dataset/{dataset_name}/negative_samples.pt'
    else:
        negs_name = f'{ROOT_DIR}/dataset/{dataset_name}/negative_samples_{num_negs}.pt'
    print(f'looking for negative edges at {negs_name}')
    if os.path.exists(negs_name):
        print('loading negatives from disk')
        train_negs = torch.load(negs_name)
    else:
        print('negatives not found on disk. Generating negatives')
        train_negs = get_ogb_train_negs(split_edge, data.edge_index, data.num_nodes, num_negs, dataset_name)
        torch.save(train_negs, negs_name)

    splits = {}
    for key in split_edge.keys():
        # the ogb datasets come with test and valid negatives, but you have to cook your own train negs
        neg_edges = train_negs if key == 'train' else None
        edge_label, edge_label_index = make_obg_supervision_edges(split_edge, key, neg_edges)
        # use the validation edges for message passing at test time
        # according to the rules https://ogb.stanford.edu/docs/leader_rules/ only collab can use val edges at test time
        if key == 'test' and dataset_name == 'ogbl-collab':
            vei, vw = to_undirected(split_edge['valid']['edge'].t(), split_edge['valid']['weight'])
            edge_index = torch.cat([data.edge_index, vei], dim=1)
            edge_weight = torch.cat([data.edge_weight, vw.unsqueeze(-1)], dim=0)
        else:
            edge_index = data.edge_index
            if hasattr(data, "edge_weight"):
                edge_weight = data.edge_weight
            else:
                edge_weight = torch.ones(data.edge_index.shape[1])
        splits[key] = Data(x=data.x, edge_index=edge_index, edge_weight=edge_weight, edge_label=edge_label,
                           edge_label_index=edge_label_index)
    return splits


def get_ogb_pos_edges(split_edge, split):
    if 'edge' in split_edge[split]:
        pos_edge = split_edge[split]['edge']
    elif 'source_node' in split_edge[split]:
        pos_edge = torch.stack([split_edge[split]['source_node'], split_edge[split]['target_node']],
                               dim=1)
    else:
        raise NotImplementedError
    return pos_edge


def get_ogb_train_negs(split_edge, edge_index, num_nodes, num_negs=1, dataset_name=None):
    """
    for some inexplicable reason ogb datasets split_edge object stores edge indices as (n_edges, 2) tensors
    @param split_edge:

    @param edge_index: A [2, num_edges] tensor
    @param num_nodes:
    @param num_negs: the number of negatives to sample for each positive
    @return: A [num_edges * num_negs, 2] tensor of negative edges
    """
    pos_edge = get_ogb_pos_edges(split_edge, 'train').t()
    if dataset_name is not None and dataset_name.startswith('ogbl-citation'):
        neg_edge = get_same_source_negs(num_nodes, num_negs, pos_edge)
    else:  # any source is fine
        new_edge_index, _ = add_self_loops(edge_index)
        neg_edge = negative_sampling(
            new_edge_index, num_nodes=num_nodes,
            num_neg_samples=pos_edge.size(1) * num_negs)
    return neg_edge.t()


def make_obg_supervision_edges(split_edge, split, neg_edges=None):
    if neg_edges is not None:
        neg_edges = neg_edges
    else:
        if 'edge_neg' in split_edge[split]:
            neg_edges = split_edge[split]['edge_neg']
        elif 'target_node_neg' in split_edge[split]:
            n_neg_nodes = split_edge[split]['target_node_neg'].shape[1]
            neg_edges = torch.stack([split_edge[split]['source_node'].unsqueeze(1).repeat(1, n_neg_nodes).ravel(),
                                     split_edge[split]['target_node_neg'].ravel()
                                     ]).t()
        else:
            raise NotImplementedError

    pos_edges = get_ogb_pos_edges(split_edge, split)
    n_pos, n_neg = pos_edges.shape[0], neg_edges.shape[0]
    edge_label = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)], dim=0)
    edge_label_index = torch.cat([pos_edges, neg_edges], dim=0).t()
    return edge_label, edge_label_index


def use_lcc(dataset):
    lcc = get_largest_connected_component(dataset)

    x_new = dataset.data.x[lcc]
    y_new = dataset.data.y[lcc]

    row, col = dataset.data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    data = Data(
        x=x_new,
        edge_index=torch.LongTensor(edges),
        y=y_new,
        train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
    )
    # original dataset._data = data
    dataset._data = data
    return dataset



def load_data(cfg, dataset, use_text, use_gpt=False, seed=0):
    if dataset == 'cora':
        from  src.data_utils.load_cora import get_raw_text_cora as get_raw_text
    elif dataset == 'pubmed':
        from  src.data_utils.load_pubmed import get_raw_text_pubmed as get_raw_text
    elif dataset == 'ogbn-arxiv':
        from  src.data_utils.load_arxiv import get_raw_text_arxiv as get_raw_text
    else:
        exit(f'Error: Dataset {dataset} not supported')

    # for training GNN
    if not use_text:
        data, _ = get_raw_text(cfg, use_text, seed=seed)
        return data

    # for finetuning LM
    if use_gpt:
        data, text = get_raw_text(cfg, use_text=False, seed=seed)
        folder_path = 'gpt_responses/{}'.format(dataset)
        print(f"using gpt: {folder_path}")
        n = data.y.shape[0]
        text = []
        for i in range(n):
            filename = str(i) + '.json'
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                content = json_data['choices'][0]['message']['content']
                text.append(content)
    else:
        data, text = get_raw_text(cfg, use_text=True, seed=seed)
    print(data)
    return data, text


class Textgraph(InMemoryDataset):
    r"""The citation network datasets :obj:`"Cora"`, :obj:`"CiteSeer"` and
    :obj:`"PubMed"` from the `"Revisiting Semi-Supervised Learning with Graph
    Embeddings" <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Cora"`, :obj:`"CiteSeer"`,
            :obj:`"PubMed"`).
        split (str, optional): The type of dataset split (:obj:`"public"`,
            :obj:`"full"`, :obj:`"geom-gcn"`, :obj:`"random"`).
            If set to :obj:`"public"`, the split will be the public fixed split
            from the `"Revisiting Semi-Supervised Learning with Graph
            Embeddings" <https://arxiv.org/abs/1603.08861>`_ paper.
            If set to :obj:`"full"`, all nodes except those in the validation
            and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"geom-gcn"`, the 10 public fixed splits from the
            `"Geom-GCN: Geometric Graph Convolutional Networks"
            <https://openreview.net/forum?id=S1e2agrFvS>`_ paper are given.
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test`. (default: :obj:`"public"`)
        num_train_per_class (int, optional): The number of training samples
            per class in case of :obj:`"random"` split. (default: :obj:`20`)
        num_val (int, optional): The number of validation samples in case of
            :obj:`"random"` split. (default: :obj:`500`)
        num_test (int, optional): The number of test samples in case of
            :obj:`"random"` split. (default: :obj:`1000`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - Cora
          - 2,708
          - 10,556
          - 1,433
          - 7
        * - CiteSeer
          - 3,327
          - 9,104
          - 3,703
          - 6
        * - PubMed
          - 19,717
          - 88,648
          - 500
          - 3
    """
    url = ''
    geom_gcn_url = ('')
    
    def __init__(self, cfg: CfgNode, name: str = 'cora', use_text: bool = False, split: str = "public",
                 num_train_per_class: int = 20, num_val: int = 500,
                 num_test: int = 1000, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        # data 
        self.name = name
        self.cfg = cfg
        self.use_text = use_text
        self.split = split
        self.dataset_name = name
        # change here to different datasets solution save it as a default variable in local file 
        self.root = dataset_path[name]['root']
        self.lm_model_name = cfg.lm.model.name 
        self.seed = cfg.seed
        self.device = cfg.device
        self.feature_type = cfg.dataset.feature_type # ogb, TA, E, P
        self.transform = transform
        self.pre_transform = pre_transform
        # split 
        assert self.split in ['public', 'full', 'geom-gcn', 'random']
        self.split = split
        self.process()

        if self.pre_transform is not None:
            self._data = self.pre_transform(self._data)
        torch.save(self.collate([self._data]), self.processed_paths[0])
        # self.data = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self) -> List[str]:
        self.prt_lm = f"{self.root}/prt_lm/{self.dataset_name}/{self.lm_model_name}-seed{self.seed}.emb"
        return [dataset_path[self.name]['original'],
        dataset_path[self.name]['papers'],
        dataset_path[self.name]['extractions'],
        self.prt_lm]

    @property 
    def processed_dir(self) -> str:
        return osp.join(self.root, f'{self.dataset_name}/processed_text') if self.use_text else osp.join(self.root, f'{self.dataset_name}/processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'


    def download(self):
        pass


    @property
    def num_nodes(self) -> int:
        return self._data.x.size(0)
    

    @property
    def num_node_features(self) -> int:
        return self._data.x.size(1) 
    

    @property 
    def num_edge_features(self) -> int:
        raise NotImplementedError
    
    @property
    def data(self) -> Any:
        return self._data

    @property
    def splits(self) -> dict:
        splits = {}
        for masks in ['train_mask', 'val_mask', 'test_mask']:
            if hasattr(self._data, masks):
                if np.sum(np.array(getattr(self._data, masks))) == 0:
                    return None
                else:
                    splits[masks] = getattr(self._data, masks)
        return splits

    def load_features(self) -> torch.Tensor:
        """generate node features

        Returns:
            torch.tensor: _description_
        """
        if self.feature_type == 'ogb':
            print("Loading OGB features...")
            self.node_feat = self._data.x

        elif self.feature_type == 'TA':
            # /pfs/work7/workspace/scratch/cc7738-prefeature/TAPE/prt_lm/cora 2/microsoft/deberta-base-seed0.ckpt
            print("Loading pretrained LM features (title and abstract) ...")
            LM_emb_path = os.path.join(self.root, f"prt_lm/{self.dataset_name}/microsoft/{self.lm_model_name}-seed{self.seed}.emb")
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                            dtype=np.float16,
                            shape=(self.num_nodes, 768)))
            ).to(torch.float32)
            self.node_feat = features

        elif self.feature_type == 'E':
            print("Loading pretrained LM features (explanations) ...")
            LM_emb_path = f"{self.root}/prt_lm/{self.dataset_name}2/microsoft/{self.lm_model_name}-seed{self.seed}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                            dtype=np.float16,
                            shape=(self.num_nodes, 768)))
            ).to(torch.float32)
            self.node_feat = features
        else:
            print(
                f'Feature type {self.feature_type} not supported. Loading OGB features...')
            self.feature_type = 'ogb'
            self.node_feat = self._data.x
    
        return self.node_feat

    def process(self):
        # read data from text files
        self._data = load_data(self.cfg, self.cfg.dataset.name, use_text=False, seed=self.seed)
        # read from pretrained LM
        self.node_feat = self.load_features()
        # replace node features
        self._data.x = self.node_feat
        # masks 
        self.get_split_mask_generator()
        # save 
        os.makedirs(self.processed_dir, exist_ok=True)
        path = osp.join(self.processed_dir, 'data.pt')
        torch.save(self._data, path)


    def print_parameters(self):
        for k, val in self.__dict__.items():
            print(k, val)

    def __repr__(self) -> str:
        return f'{self.name}()'


    def get_split_mask_generator(self):
        # availability test 
        for masks in ['train_mask', 'val_mask', 'test_mask']:
            if hasattr(self._data, masks):
                return

        if self.split == 'full':
            raise NotImplementedError
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif self.split == 'random':
            raise NotImplementedError
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])

    @property 
    def prt_lm(self):
        return f"{self.root}/prt_lm/{self.dataset_name}/{self.lm_model_name}-seed{self.seed}.emb"

    @property
    def raw_file_names(self) -> List[str]:
        return [self.cfg.dataset.cora.original,
        self.cfg.dataset.cora.papers,
        self.cfg.dataset.cora.extractions,
        self.prt_lm]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    @property
    def num_nodes(self) -> int:
        return self.data.x.size(0)

    # def process(self):
    #     # read data from text files
    #     print("Loading pretrained LM features (title and abstract) ...")
    #     data = load_data(self.cfg, self.dataset_name, use_text=False, seed=self.seed)
    #     print(f"LM_emb_path: {self.prt_lm}")
    #     features = torch.from_numpy(np.array(
    #         np.memmap(self.prt_lm, mode='r',
    #                   dtype=np.float16,
    #                   shape=(self.num_nodes, 768)))
    #     ).to(torch.float32)
    #     print(features.shape)

    #     # split masks for geom-gcn
    #     if self.split == 'geom-gcn':
    #         train_masks, val_masks, test_masks = [], [], []
    #         for i in range(10):
    #             name = None
    #             splits = None
    #             train_masks = None
    #             val_masks = None
    #             test_masks = None
    #         data.train_mask = torch.stack(train_masks, dim=1)
    #         data.val_mask = torch.stack(val_masks, dim=1)
    #         data.test_mask = torch.stack(test_masks, dim=1)

    #     data = data if self.pre_transform is None else self.pre_transform(data)
    #     self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'

