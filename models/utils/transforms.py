import torch
from torch_geometric.utils import to_dense_adj

class EyeTransform(object):
    r"""Adds the node degree as one hot encodings to the node features.

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """
    def __call__(self, data):
        n = data.x.shape[0]
        data.x = torch.eye(n, dtype=torch.float)
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