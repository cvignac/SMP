import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from models.utils.layers import XtoX, UtoU, UtoU, EntrywiseU, EntryWiseX


class SimplifiedFastSMPLayer(MessagePassing):
    def __init__(self, in_features: int, num_towers: int, out_features: int, use_x: bool):
        super().__init__(aggr='add', node_dim=-3)
        self.use_x = use_x
        self.message_nn = (XtoX if use_x else UtoU)(in_features, out_features, bias=True)
        if self.use_x:
            self.alpha = nn.Parameter(torch.zeros(1, out_features), requires_grad=True)
        else:
            self.alpha = nn.Parameter(torch.zeros(1, 1, out_features), requires_grad=True)

    def reset_parameters(self):
        self.message_nn.reset_parameters()
        self.alpha.requires_grad_(False)
        self.alpha[...] = 0
        self.alpha.requires_grad_(True)

    def forward(self, u, edge_index, batch_info):
        """ x corresponds either to node features or to the local context, depending on use_x."""
        n = batch_info['num_nodes']
        if self.use_x and u.dim() == 1:
            u = u.unsqueeze(-1)
        u = self.message_nn(u, batch_info)
        new_u = self.propagate(edge_index, size=(n, n), u=u)
        # Normalization
        if len(new_u.shape) == 2:
            # node features are used
            new_u /= batch_info['average_edges'][:, :, 0]
        else:
            # local contexts are used
            new_u /= batch_info['average_edges']
        return new_u

    def message(self, u_j: Tensor):
        return u_j

    def update(self, aggr_u, u):
        return aggr_u + u + self.alpha * u * aggr_u


class FastSMPLayer(MessagePassing):
    def __init__(self, in_features: int, num_towers: int, out_features: int, use_x: bool):
        super().__init__(aggr='add', node_dim=-3)
        self.use_x = use_x
        self.in_u, self.out_u = in_features, out_features
        if use_x:
            self.message_nn = XtoX(in_features, out_features, bias=True)
            self.linu_i = EntryWiseX(out_features, out_features, num_towers)
            self.linu_j = EntryWiseX(out_features, out_features, num_towers)
        else:
            self.message_nn = UtoU(in_features, out_features, n_groups=num_towers, residual=False)
            self.linu_i = EntrywiseU(out_features, out_features, num_towers=num_towers)
            self.linu_j = EntrywiseU(out_features, out_features, num_towers=num_towers)

    def forward(self, u, edge_index, batch_info):
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
        return aggr_u + u + a_i * a_j


class SMPLayer(MessagePassing):
    def __init__(self, in_features: int, num_towers: int, out_features: int, use_x: bool):
        super().__init__(aggr='add', node_dim=-3)
        self.use_x = use_x
        self.in_u, self.out_u = in_features, out_features
        if use_x:
            self.message_nn = XtoX(in_features, out_features, bias=True)
            self.order2_i = EntryWiseX(out_features, out_features, num_towers)
            self.order2_j = EntryWiseX(out_features, out_features, num_towers)
            self.order2 = EntryWiseX(out_features, out_features, num_towers)
        else:
            self.message_nn = UtoU(in_features, out_features, n_groups=num_towers, residual=False)
            self.order2_i = EntrywiseU(out_features, out_features, num_towers)
            self.order2_j = EntrywiseU(out_features, out_features, num_towers)
            self.order2 = EntrywiseU(out_features, out_features, num_towers)
        self.update1 = nn.Linear(2 * out_features, out_features)
        self.update2 = nn.Linear(out_features, out_features)

    def forward(self, u, edge_index, batch_info):
        n = batch_info['num_nodes']
        u = self.message_nn(u, batch_info)
        u1 = self.order2_i(u)
        u2 = self.order2_j(u)
        new_u = self.propagate(edge_index, size=(n, n), u=u, u1=u1, u2=u2)
        new_u /= batch_info['average_edges']
        return new_u

    def message(self, u_j, u1_i, u2_j):
        order2 = self.order2(torch.relu(u1_i + u2_j))
        return order2
        # u_j = u_j + order2
        # return u_j

    def update(self, aggr_u, u):
        up1 = self.update1(torch.cat((u, aggr_u), dim=-1))
        up2 = up1 + self.update2(up1)
        return up2


class ZincSMPLayer(MessagePassing):
    def __init__(self, in_features: int, num_towers: int, out_features: int, edge_features: int, use_x: bool):
        """ Use a MLP both for the update and message function + edge features"""
        super().__init__(aggr='add', node_dim=-3)
        self.use_x = use_x
        self.in_u, self.out_u, self.edge_features = in_features, out_features, edge_features
        self.edge_nn = nn.Linear(edge_features, out_features)
        if use_x:
            self.message_nn = XtoX(in_features, out_features, bias=True)
            self.order2_i = EntryWiseX(out_features, out_features, num_towers)
            self.order2_j = EntryWiseX(out_features, out_features, num_towers)
            self.order2 = EntryWiseX(out_features, out_features, num_towers)
        else:
            self.message_nn = UtoU(in_features, out_features, n_groups=num_towers, residual=False)
            self.order2_i = EntrywiseU(out_features, out_features, num_towers)
            self.order2_j = EntrywiseU(out_features, out_features, num_towers)
            self.order2 = EntrywiseU(out_features, out_features, num_towers)
        self.update1 = nn.Linear(2 * out_features, out_features)
        self.update2 = nn.Linear(out_features, out_features)

    def forward(self, u, edge_index, edge_attr, batch_info):
        n = batch_info['num_nodes']
        u = self.message_nn(u, batch_info)
        u1 = self.order2_i(u)
        u2 = self.order2_j(u)
        new_u = self.propagate(edge_index, size=(n, n), u=u, u1=u1, u2=u2, edge_attr=edge_attr)
        new_u /= batch_info['average_edges']
        return new_u

    def message(self, u_j, u1_i, u2_j, edge_attr):
        edge_feat = self.edge_nn(edge_attr).unsqueeze(1)
        order2 = self.order2(torch.relu(u1_i + u2_j + edge_feat))
        u_j = u_j + order2
        return u_j

    def update(self, aggr_u, u):
        up1 = self.update1(torch.cat((u, aggr_u), dim=-1))
        up2 = up1 + self.update2(up1)
        return up2 + u


class FilmZincSMPLayer(MessagePassing):
    def __init__(self, in_features: int, num_towers: int, out_features: int, edge_features: int, use_x: bool):
        """ Use a MLP both for the update and message function + edge features"""
        super().__init__(aggr='add', node_dim=-3)
        self.use_x = use_x
        self.in_u, self.out_u, self.edge_features = in_features, out_features, edge_features
        self.alpha = nn.Linear(edge_features, out_features)
        self.beta = nn.Linear(edge_features, out_features)
        if use_x:
            self.message_nn = XtoX(in_features, out_features, bias=True)
            self.order2_i = EntryWiseX(out_features, out_features, num_towers)
            self.order2_j = EntryWiseX(out_features, out_features, num_towers)
            self.order2 = EntryWiseX(out_features, out_features, num_towers)
        else:
            self.message_nn = UtoU(in_features, out_features, n_groups=num_towers, residual=False)
            self.order2_i = EntrywiseU(out_features, out_features, num_towers)
            self.order2_j = EntrywiseU(out_features, out_features, num_towers)
            self.order2 = EntrywiseU(out_features, out_features, num_towers)
        self.update1 = nn.Linear(2 * out_features, out_features)
        self.update2 = nn.Linear(out_features, out_features)

    def forward(self, u, edge_index, edge_attr, batch_info):
        n = batch_info['num_nodes']
        u = self.message_nn(u, batch_info)
        u1 = self.order2_i(u)
        u2 = self.order2_j(u)
        new_u = self.propagate(edge_index, size=(n, n), u=u, u1=u1, u2=u2, edge_attr=edge_attr)
        new_u /= batch_info['average_edges']
        return new_u

    def message(self, u_j, u1_i, u2_j, edge_attr):
        alpha = self.alpha(edge_attr).unsqueeze(1) + 1
        beta = self.beta(edge_attr).unsqueeze(1)
        order2 = self.order2(torch.relu(u1_i + u2_j))
        u_j = u_j + order2
        u_j = alpha * u_j + beta
        return u_j

    def update(self, aggr_u, u):
        up1 = self.update1(torch.cat((u, aggr_u), dim=-1))
        up2 = up1 + self.update2(up1)
        return up2