from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
# param

from config_load import cfg_data as cfg
from config_load import update_cfg
from  pdb import set_trace as bp

def get_raw_text_arxiv(cfg, use_text=False, seed=0):
    # dataset = PygNodePropPredDataset(
    #     name='ogbn-arxiv', transform=T.ToSparseTensor())
    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv')
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None

    nodeidx2paperid = pd.read_csv(
        'dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')

    raw_text = pd.read_csv(cfg.dataset.arxiv.abs_ti,
                           sep='\t', header=None, names=['paper id', 'title', 'abs'])

    # remove string paper id
    nodeidx2paperid['paper id'] = nodeidx2paperid['paper id'].astype(int)
    raw_text = raw_text.dropna()
    raw_text.loc[1:, 'paper id'] = raw_text[1:]['paper id'].astype(int)
    df = pd.merge(nodeidx2paperid, raw_text[1:], on='paper id')
    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = 'Title: ' + ti + '\n' + 'Abstract: ' + ab
        text.append(t)
    return data, text


if __name__ == '__main__':
    cfg = update_cfg(cfg)

    data, text = get_raw_text_arxiv(cfg, use_text=True)
    print(data)
    print(text)
