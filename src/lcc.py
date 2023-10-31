"""
utils for getting the largest connected component of a graph
"""
import numpy as np
from torch_geometric.data import Data, InMemoryDataset

def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
  remaining_nodes = set(range(dataset.data.x.shape[0]))
  print(f"remaining_nodes: {len(remaining_nodes)}")
  comps = []
  while remaining_nodes:
    start = min(remaining_nodes)
    comp = get_component(dataset, start)
    print(len(comp))
    comps.append(comp)
    remaining_nodes = remaining_nodes.difference(comp)
  return np.array(list(comps[np.argmax(list(map(len, comps)))]))

import scipy.sparse as sp
def largest_connected_components(sparse_graph: 'gust.SparseGraph', n_components: int = 1) -> 'gust.SparseGraph':
    """Select the largest connected components in the graph.

    Changes are returned in a partially new SparseGraph.

    Parameters
    ----------
    sparse_graph
        Input graph.
    n_components
        Number of largest connected components to keep.

    Returns
    -------
    gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    # TODO: add warnings / logging
    # print("Selecting {0} largest connected components".format(n_components))
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)

def get_node_mapper(lcc: np.ndarray) -> dict:
  mapper = {}
  counter = 0
  for node in lcc:
    mapper[node] = counter
    counter += 1
  return mapper


def remap_edges(edges: list, mapper: dict) -> list:
  row = [e[0] for e in edges]
  col = [e[1] for e in edges]
  row = list(map(lambda x: mapper[x], row))
  col = list(map(lambda x: mapper[x], col))
  return [row, col]

def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
  visited_nodes = set()
  queued_nodes = set([start])
  row, col = dataset.data.edge_index.numpy()
  while queued_nodes:
    current_node = queued_nodes.pop()
    visited_nodes.update([current_node])
    neighbors = col[np.where(row == current_node)[0]]
    neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
    queued_nodes.update(neighbors)
  return visited_nodes