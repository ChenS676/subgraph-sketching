import os
import yaml
import torch
import math
import numpy as np

def extract_eval(lst):
    """
    extract the formula as string from a list into real code
    e.g.
    from:
        [0, '1/math.log(2)', 0, 0]
    to:
        [0., 1.44269504, 0., 0.]
    """
    lst = list(eval(s) if isinstance(s,str) else s for s in lst)
    return lst

def get_graph_config(config_yaml,show=False):
    """
    get pre-selected features for creation of dataset, including distance and angle features
    """
    config_yaml = os.path.join('./test/graph_configs',config_yaml)
    output_config = {}
    with open(config_yaml, "r") as file:
        load_config = yaml.safe_load(file)
    output_config['algorithm'] = load_config['algorithm'] # loaded: string
    output_config['num_nodes'] = load_config['num_nodes'] # loaded: int
    output_config['edge_index'] = torch.tensor(load_config['edge_index']).t() # loaded: list probably with string which needs eval()
    output_config['pos_test_edges'] = torch.tensor(load_config['pos_test_edges']).t()
    output_config['neg_test_edges'] = torch.tensor(load_config['neg_test_edges']).t()
    output_config['train_scores_target'] = np.array(extract_eval(load_config['train_scores_target']))
    output_config['pos_scores_target'] = np.array(extract_eval(load_config['pos_scores_target']))
    output_config['neg_scores_target'] = np.array(extract_eval(load_config['neg_scores_target']))
    del load_config
    if show:
        for k,v in output_config.items():
            print(f'{k}:{v}')
    return output_config

if __name__ == '__main__':

    config_yaml = 'DefaultExp_AA.yaml'
    output_config = get_graph_config(config_yaml,show=True)
    print(type(output_config['neg_scores_target']))