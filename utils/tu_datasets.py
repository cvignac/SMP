import os.path as osp

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from models.utils.transforms import KHopColoringTransform


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset(name, k, sparse=True, cleaned=False):
    if osp.isdir('/datasets2/'):
        rootdir = '/datasets2/TU_DATASETS/'
        path = osp.join(rootdir, name)
    else:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        print("Warning: local dataset was stored locally")

    transform = None
    if name in ['REDDIT-BINARY', 'REDDIT-5K']:
        transform = KHopColoringTransform(k)
    else:
        transform = None

    dataset = TUDataset(path, name, transform, cleaned=cleaned)
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)
    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

    return dataset