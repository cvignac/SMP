import torch
from torch_geometric.utils import to_dense_adj
import networkx as nx
import torch_geometric
from torch_geometric.data import Data
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path_length
from networkx.algorithms.coloring import greedy_color
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class EyeTransform(object):
    def __init__(self, max_num_nodes):
        self.max_num_nodes = max_num_nodes

    def __call__(self, data):
        n = data.x.shape[0]
        data.x = torch.eye(n, self.max_num_nodes, dtype=torch.float)
        return data

    def __repr__(self):
        return str(self.__class__.__name__)


class RandomId(object):
    r"""Adds the node degree as one hot encodings to the node features.

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """
    def __init__(self):
        pass
    def __call__(self, data):
        n = data.x.shape[0]
        data.x = torch.randint(0, 100, (n, 1), dtype=torch.float) / 100
        # data.x = torch.randn(n, self.embedding_size, dtype=torch.float)
        return data

    def __repr__(self):
        return str(self.__class__.__name__)


class DenseAdjMatrix(object):
    def __init__(self, n: int):
        """ n: number of nodes in  the graph (should be constant)"""
        self.n = n

    def __call__(self, data):
        batch = data.edge_index.new_zeros(self.n)
        data.A = to_dense_adj(data.edge_index, batch)
        return data

    def __repr__(self):
        return str(self.__class__.__name__)


class KHopColoringTransform(object):
    def __init__(self, k: int):
        self.k = k

    def __call__(self, data):
        """ Compute a coloring such that no node sees twice the same color in its k-hop neighbourhood."""
        k = self.k
        g = torch_geometric.utils.to_networkx(data, to_undirected=True, remove_self_loops=True)
        lengths = all_pairs_shortest_path_length(g, cutoff=2 * k)
        lengths = [l for l in lengths]
        # Graph where 2k hop neighbors are connected
        k_hop_graph = nx.Graph()
        for lengths_tuple in lengths:
            origin = lengths_tuple[0]
            edges = [(origin, dest) for dest in lengths_tuple[1].keys()]
            k_hop_graph.add_edges_from(edges)
        # Color the k-hop graph
        best_n_colors = np.infty
        best_color_dict = None
        # for strategy in ['largest_first', 'random_sequential', 'saturation_largest_first']:
        for strategy in ['largest_first']:
            color_dict = greedy_color(k_hop_graph, strategy)
            n_colors = np.max([color for color in color_dict.values()]) + 1
            if n_colors < best_n_colors:
                best_n_colors = n_colors
                best_color_dict = color_dict
        # Convert back to torch-geometric. The coloring is contained in data.x
        data.coloring = torch.zeros((data.num_nodes, 1), dtype=torch.long)
        for key, val in best_color_dict.items():
            data.coloring[key] = val
        print('Number of nodes: {} - Number of colors: {}'.format(data.num_nodes, data.coloring.max() + 1))
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.k)


class OneHotNodeEdgeFeatures(object):
    def __init__(self, node_types, edge_types):
        self.c = node_types
        self.d = edge_types

    def __call__(self, data):
        n = data.x.shape[0]
        node_encoded = torch.zeros((n, self.c), dtype=torch.float32)
        node_encoded.scatter_(1, data.x.long(), 1)
        data.x = node_encoded
        e = data.edge_attr.shape[0]
        edge_encoded = torch.zeros((e, self.d), dtype=torch.float32)
        edge_attr = (data.edge_attr - 1).long().unsqueeze(-1)
        edge_encoded.scatter_(1, edge_attr, 1)
        data.edge_attr = edge_encoded
        return data

    def __repr__(self):
        return str(self.__class__.__name__)