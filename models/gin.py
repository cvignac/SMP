import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from models.utils.layers import XtoGlobal


class FeatureExtractor(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.XtoG = XtoGlobal(in_features, out_features, bias=True)
        self.lin = Linear(out_features, out_features, bias=False)

    def forward(self, x, batch_info):
        """ x:  (num_nodes, in_features)
            output: (batch_size, out_features). """
        out = self.XtoG.forward(x, batch_info)
        out = out + self.lin.forward(F.relu(out))
        return out


class GINNetwork(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin_1 = nn.Linear(in_features, in_features)
        self.lin_2 = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.lin_2(x + torch.relu(self.lin_1(x)))
        return x


class GIN(nn.Module):
    def __init__(self, num_input_features: int, num_classes: int, num_layers: int,
                 hidden, hidden_final: int, dropout_prob: float, use_batch_norm: bool):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.dropout_prob = dropout_prob
        self.no_prop = FeatureExtractor(num_input_features, hidden_final)
        self.initial_lin_x = nn.Linear(num_input_features, hidden)

        self.convs = nn.ModuleList([])
        self.batch_norm_x = nn.ModuleList()
        self.feature_extractors = nn.ModuleList([])
        for i in range(num_layers):
            self.convs.append(GINConv(GINNetwork(hidden, hidden)))
            self.feature_extractors.append(FeatureExtractor(hidden, hidden_final))
            self.batch_norm_x.append(nn.BatchNorm1d(hidden))

        self.after_conv = nn.Linear(hidden_final, hidden_final)
        self.final_lin = nn.Linear(hidden_final, num_classes)

    def forward(self, data):
        """ data.x: (num_nodes, num_features)"""
        x, edge_index, batch, batch_size = data.x, data.edge_index, data.batch, data.num_graphs

        # Compute some information about the batch
        # Count the number of nodes in each graph
        unique, n_per_graph = torch.unique(data.batch, return_counts=True)
        n_batch = torch.zeros_like(batch, dtype=torch.float)

        for value, n in zip(unique, n_per_graph):
            n_batch[batch == value] = n.float()

        # Aggregate into a dict
        batch_info = {'num_nodes': data.num_nodes,
                      'num_graphs': data.num_graphs,
                      'batch': data.batch}

        out = self.no_prop.forward(x, batch_info)
        x = self.initial_lin_x(x)
        for i, (conv, bn_x, extractor) in enumerate(zip(self.convs, self.batch_norm_x, self.feature_extractors)):
            if self.use_batch_norm and i > 0:
                x = bn_x(x)
            x = conv(x, edge_index)
            global_features = extractor.forward(x, batch_info)
            out += global_features

        out = F.relu(out) / len(self.convs)
        out = F.relu(self.after_conv(out)) + out
        out = F.dropout(out, p=self.dropout_prob, training=self.training)
        out = self.final_lin(out)
        return F.log_softmax(out, dim=-1)

    def __repr__(self):
        return self.__class__.__name__