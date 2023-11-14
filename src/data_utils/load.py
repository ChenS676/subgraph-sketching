import os, json
from yacs.config import CfgNode as CN

def set_data_cfg(cfg):

	# ------------------------------------------------------------------------ #
	# Basic options
	# ------------------------------------------------------------------------ #
	# Cuda device number, used for machine with multiple gpus
	cfg.device = 0
	# Whether fix the running seed to remove randomness
	cfg.seed = 1

	cfg.dataset = CN()
	cfg.dataset.name = 'cora'	
	# root of dataset 

	cfg.dataset.cora = CN()
	cfg.dataset.cora.root = '/pfs/work7/workspace/scratch/cc7738-prefeature/TAPE' 
	cfg.dataset.cora.original = cfg.dataset.cora.root + '/dataset/cora_orig/cora'
	cfg.dataset.cora.papers =  cfg.dataset.cora.root + '/dataset/cora_orig/mccallum/cora/papers'
	cfg.dataset.cora.extractions =  cfg.dataset.cora.root + '/dataset/cora_andrew_mccallum/extractions/'
	cfg.dataset.cora.lm_model_name = 'microsoft/deberta-base'
	# ------------------------------------------------------------------------ #
	cfg.dataset.pubmed = CN()
	cfg.dataset.pubmed.root = '/pfs/work7/workspace/scratch/cc7738-prefeature/TAPE' 
	cfg.dataset.pubmed.original = cfg.dataset.pubmed.root  + '/dataset/PubMed_orig/data/'
	cfg.dataset.pubmed.abs_ti = cfg.dataset.pubmed.root  + '/dataset/PubMed_orig/pubmed.json' 

	cfg.dataset.arxiv = CN()
	cfg.dataset.arxiv.root = '/pfs/work7/workspace/scratch/cc7738-prefeature/TAPE'
	cfg.dataset.arxiv.abs_ti = cfg.dataset.arxiv.root + '/dataset/ogbn_arxiv_orig/titleabs.tsv'

	cfg.dataset.feature_type = 'TA'
	return cfg


"""
    Global variable
"""
cfg = set_data_cfg(CN())

def get_text_graph(dataset: str, 
                   use_text: bool = False, 
                   use_gpt : bool =False, 
                   seed: int =0):
    if dataset == 'cora':
        from load_cora import get_raw_text_cora as get_raw_text
    elif dataset == 'pubmed':
        from load_pubmed import get_raw_text_pubmed as get_raw_text
    elif dataset == 'ogbn-arxiv':
        from load_arxiv import get_raw_text_arxiv as get_raw_text
    else:
        exit(f'Error: Dataset {dataset} not supported')

    # for training GNN
    if not use_text:
        data, _ = get_raw_text(cfg, use_text=False, seed=seed)
        return data
    else:# for finetuning LM
        data, text = get_raw_text(cfg, use_text=True, seed=seed)
        return data, text

if __name__ == '__main__':
    data, text = get_text_graph('cora', use_text=True) 
    print(data)
    print(text)
    data, text = get_text_graph('cora', use_text=True) 
    print(data)
    print(text)
    data, text = get_text_graph('cora', use_text=True) 
    print(data)
    print(text)