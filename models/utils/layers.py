import torch
import torch.nn as nn
from torch import Tensor as Tensor
from torch.nn import Linear as Linear
import torch.nn.init as init
from torch.nn.init import _calculate_correct_fan, calculate_gain
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, MessagePassing
import math

small_gain = 0.01


def pooling(x: torch.Tensor, batch_info, method):
    if method == 'add':
        return global_add_pool(x, batch_info['batch'], batch_info['num_graphs'])
    elif method == 'mean':
        return global_mean_pool(x, batch_info['batch'], batch_info['num_graphs'])
    elif method == 'max':
        return global_max_pool(x, batch_info['batch'], batch_info['num_graphs'])
    else:
        raise ValueError("Pooling method not implemented")


def kaiming_init_with_gain(x: Tensor, gain: float, a=0, mode='fan_in', nonlinearity='relu'):
    fan = _calculate_correct_fan(x, mode)
    non_linearity_gain = calculate_gain(nonlinearity, a)
    std = non_linearity_gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std * gain   # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return x.uniform_(-bound, bound)


class BatchNorm(nn.Module):
    def __init__(self, channels: int, use_x: bool):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels)
        self.use_x = use_x

    def forward(self, u):
        if self.use_x:
            return self.bn(u)
        else:
            return self.bn(u.transpose(1, 2)).transpose(1, 2)


class EdgeCounter(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, batch, batch_size):
        n_edges = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        return global_mean_pool(n_edges, batch, batch_size)[batch]


class Linear(nn.Module):
    """ Linear layer with potentially smaller parameters at initialization. """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, gain: float = 1.0):
        super().__init__()
        self.gain = gain
        self.lin = nn.Linear(in_features, out_features, bias)

    def reset_parameters(self):
        kaiming_init_with_gain(self.lin.weight, self.gain)
        if self.lin.bias is not None:
            nn.init.normal_(self.lin.bias, 0, self.gain / math.sqrt(self.lin.out_features))

    def forward(self, x):
        return self.lin.forward(x)


class XtoX(Linear):
    def forward(self, x, batch_info: dict = None):
        return self.lin.forward(x)


class XtoGlobal(Linear):
    def forward(self, x: Tensor, batch_info: dict, method='mean'):
        """ x: (num_nodes, in_features). """
        g = pooling(x, batch_info, method)  # bs, N, in_feat or bs, in_feat
        return self.lin.forward(g)


class ChannelWiseU(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_groups=None):
        super().__init__()
        if n_groups is None:
            n_groups = in_features
        self.lin1 = torch.nn.Conv1d(in_features, out_features, kernel_size=1, groups=n_groups, bias=False)

    def forward(self, u):
        """ u: N x colors x channels. """
        u = u.transpose(1, 2)
        u = self.lin1(u)
        return u.transpose(1, 2)


class UtoU(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, gain: float = 1):
        super().__init__()
        self.lin1 = Linear(in_features, out_features, bias, gain)
        self.lin2 = Linear(in_features, out_features, False, 0.1 * gain)
        self.lin3 = Linear(in_features, out_features, False, 0.1 * gain)

    def forward(self, u: Tensor, batch_info: dict):
        n = batch_info['num_nodes']
        num_colors = u.shape[1]
        out_feat = self.lin1.lin.out_features
        mask = batch_info['mask'][..., None].expand(n, num_colors, out_feat)
        normalizer = batch_info['n_batch']
        mean2 = torch.sum(u / normalizer, dim=1)     # N, in_feat
        # 1. Transform u element-wise
        out = self.lin1.lin.forward(u)                   # N, n_colors, out_feat

        # 2. Put in self of each line the sum over each line
        z2 = self.lin2.lin.forward(mean2)                # N, out_feat
        z = z2[:, None, :]                                    # N, 1, out_feat
        index_tensor = batch_info['coloring'][:, :, None].expand(out.shape[0], 1, out_feat)
        out.scatter_add_(1, index_tensor, z)      # n, n_colors, out_feat

        # 3. Put everywhere the sum over each line
        z3 = self.lin3.lin.forward(mean2)[:, None, :]         # N, out_feat
        out3 = z3.expand(n, num_colors, out_feat)
        out += out3 * mask
        return out


class SimpleUtoU(nn.Module):
    def __init__(self, in_features: int, out_features: int, residual=True, n_groups=None):
        super().__init__()
        if n_groups is None:
            n_groups = out_features
        self.residual = residual
        self.lin1 = torch.nn.Conv1d(in_features, out_features, kernel_size=1, groups=n_groups, bias=True)
        self.lin2 = torch.nn.Conv1d(in_features, out_features, kernel_size=1, groups=n_groups, bias=True)
        self.lin3 = torch.nn.Conv1d(in_features, out_features, kernel_size=1, groups=n_groups, bias=True)

    def forward(self, u: Tensor, batch_info: dict = None):
        """ U: N x n_colors x channels"""
        old_u = u
        n = batch_info['num_nodes']
        num_colors = u.shape[1]
        out_feat = self.lin1.out_channels
        mask = batch_info['mask'][..., None].expand(n, num_colors, out_feat)
        normalizer = batch_info['n_batch']
        mean2 = torch.sum(u / normalizer, dim=1)     # N, in_feat
        mean2 = mean2.unsqueeze(-1)                  # N, in_feat, 1
        # 1. Transform u element-wise
        u = u.permute(0, 2, 1)
        out = self.lin1(u).permute(0, 2, 1)

        # 2. Put in self of each line the sum over each line
        z2 = 0.1 * self.lin2(mean2)                            # N, out_feat, 1
        z2 = z2.transpose(1, 2)                          # N, 1, out_feat
        index_tensor = batch_info['coloring'][:, :, None].expand(out.shape[0], 1, out_feat)
        out.scatter_add_(1, index_tensor, z2)      # n, n_colors, out_feat

        # 3. Put everywhere the sum over each line
        z3 = 0.1 * self.lin3(mean2)                       # N, out_feat, 1
        z3 = z3.transpose(1, 2)                     # N, 1, out_feat
        out3 = z3.expand(n, num_colors, out_feat)
        out += out3 * mask                          # Mask the extra colors
        if self.residual:
            return old_u + out
        return out


class UtoGlobal(nn.Module):
    def __init__(self, in_features: int , out_features: int, bias: bool, gain: float):
        super().__init__()
        self.lin1 = Linear(in_features, out_features, bias, gain=gain)
        self.lin2 = Linear(in_features, out_features, bias, gain=gain)

    def forward(self, u, batch_info: dict, method='mean'):
        """ u: (num_nodes, colors, in_features)
            output: (batch_size, out_features). """
        coloring = batch_info['coloring']
        # Extract trace
        index_tensor = coloring[:, :, None].expand(u.shape[0], 1, u.shape[2])
        extended_diag = u.gather(1, index_tensor)[:, 0, :]          # n_nodes, in_feat
        mean_batch_trace = pooling(extended_diag, batch_info, 'mean')    # n_graphs, in_feat
        out1 = self.lin1(mean_batch_trace)                   # bs, out_feat
        # Extract sum of elements - trace
        mean = torch.sum(u / batch_info['n_batch'], dim=1)  # num_nodes, in_feat
        batch_sum = pooling(mean, batch_info, 'mean')                    # n_graphs, in_feat
        batch_sum = batch_sum - mean_batch_trace                         # make the basis orthogonal
        out2 = self.lin2(batch_sum)  # bs, out_feat
        return out1 + out2


class NodeExtractor(nn.Module):
    def __init__(self, in_features_u: int, out_features_u: int):
        super().__init__()
        # Extract from U with a Deep set
        self.lin1_u = nn.Linear(in_features_u, in_features_u)
        self.lin2_u = nn.Linear(in_features_u, in_features_u)
        self.combine1 = nn.Linear(3 * in_features_u, out_features_u)

    def forward(self, x: Tensor, u: Tensor, batch_info: dict):
        """ u: (num_nodes, num_nodes, in_features).
            output: (num_nodes, out_feat).
            this method can probably be made more efficient.
        """
        # Extract u
        new_u = self.lin2_u(torch.relu(self.lin1_u(u)))
        # Aggregation
        # a. Extract the value in self
        index_tensor = batch_info['coloring'][:, :, None].expand(u.shape[0], 1, u.shape[-1])
        x1 = torch.gather(new_u, 1, index_tensor)
        x1 = x1[:, 0, :]
        # b. Mean over the line
        x2 = torch.sum(new_u / batch_info['n_batch'], dim=1)  # num_nodes x in_feat
        # c. Max over the line
        x3 = torch.max(new_u, dim=1)[0]  # num_nodes x out_feat
        # Combine
        x_full = torch.cat((x1, x2, x3), dim=1)
        out = self.combine1(x_full)
        return out


class GraphExtractor(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_x: bool):
        super().__init__()
        self.use_x = use_x
        self.extractor = (XtoGlobal if self.use_x else UtoGlobal)(in_features, out_features, True, 1)
        self.lin = nn.Linear(out_features, out_features)

    def forward(self, u: Tensor, batch_info: dict):
        out = self.extractor(u, batch_info)
        out = out + self.lin(F.relu(out))
        return out
