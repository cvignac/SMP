import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from models.utils.layers import XtoX, UtoU, SimpleUtoU, ChannelWiseU


class SimpleTypeASMPLayer(MessagePassing):
    def __init__(self, in_features: int, out_features: int, use_x: bool, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.use_x = use_x
        self.message_nn = (XtoX if use_x else UtoU)(in_features, out_features, bias=True)
        if self.use_x:
            self.alpha = nn.Parameter(torch.zeros(1, out_features), requires_grad=True)
        else:
            self.alpha = nn.Parameter(torch.zeros(1, 1, out_features), requires_grad=True)

    def forward(self, u, edge_index, batch_info):
        """ x corresponds either to node features or to the local context, depending on use_x."""
        n = batch_info['num_nodes']
        if self.use_x and u.dim() == 1:
            u = u.unsqueeze(-1)
        u = self.message_nn(u, batch_info)
        new_u = self.propagate(edge_index, size=(n, n), u=u)
        new_u /= (batch_info['average_edges'][:, :, 0] if self.use_x else batch_info['average_edges'])
        return new_u

    def message(self, u_j: Tensor):
        return u_j

    def update(self, aggr_u, u):
        aggr_u = aggr_u + self.alpha * u * aggr_u
        return aggr_u + u


class TypeASMPLayer(MessagePassing):
    def __init__(self, in_features: int, num_towers: int,
                 out_features: int, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.in_u, self.out_u = in_features, out_features
        self.message_nn = SimpleUtoU(in_features, out_features, n_groups=num_towers, residual=False)
        self.linu_i = ChannelWiseU(out_features, out_features, num_towers)
        self.linu_j = ChannelWiseU(out_features, out_features, num_towers)

    def forward(self, u, edge_index, batch_info, debug_model):
        n = batch_info['num_nodes']
        u = self.message_nn(u, batch_info)
        new_u = self.propagate(edge_index, size=(n, n), u=u)
        new_u /= batch_info['average_edges']
        return new_u

    def message(self, u_j):
        return u_j

    def update(self, aggr_u, u):
        a_i = self.linu_i(u)
        a_j = self.linu_j(aggr_u)
        aggr_u = (aggr_u + a_i * a_j) / 2
        return aggr_u

class TypeBSMPLayer(MessagePassing):
    def __init__(self, in_features: int, num_towers: int,
                 out_features: int, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.in_u, self.out_u = in_features, out_features
        self.message_nn = SimpleUtoU(in_features, out_features, n_groups=num_towers, residual=False)
        self.order2_i = ChannelWiseU(out_features, out_features, num_towers)
        self.order2_j = ChannelWiseU(out_features, out_features, num_towers)
        self.order2 = ChannelWiseU(out_features, out_features, num_towers)

    def forward(self, u, edge_index, batch_info, debug_model):
        n = batch_info['num_nodes']
        u = self.message_nn(u, batch_info)
        u1 = self.order2_i(u)
        u2 = self.order2_j(u)
        new_u = self.propagate(edge_index, size=(n, n), u=u, u1=u1, u2=u2)
        new_u /= batch_info['average_edges']
        return new_u

    def message(self, u_j, u1_i, u2_j):
        order2 = self.order2(torch.relu(u1_i + u2_j))
        u_j = u_j + order2
        return u_j

    def update(self, aggr_u):
        return aggr_u