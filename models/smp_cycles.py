import torch
import torch.nn.functional as F
import torch.nn as nn
from models.smp_layers import SimpleTypeASMPLayer
from models.utils.layers import GraphExtractor, EdgeCounter, BatchNorm
from models.utils.misc import create_batch_info


class SMP(torch.nn.Module):
    def __init__(self, num_input_features: int, num_classes: int, num_layers: int, hidden: int,
                 hidden_final: int, dropout_prob: float, use_batch_norm: bool, use_x: bool,
                 simplified: bool):
        super().__init__()
        self.use_x = use_x
        self.dropout_prob = dropout_prob
        self.use_batch_norm = use_batch_norm
        self.edge_counter = EdgeCounter()
        if not self.use_x:
            num_input_features = 1
        print("Use x:", use_x)

        self.no_prop = GraphExtractor(in_features=num_input_features, out_features=hidden_final, use_x=use_x)
        self.initial_lin = nn.Linear(num_input_features, hidden)

        self.convs = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        self.feature_extractors = torch.nn.ModuleList([])
        for i in range(0, num_layers):
            self.convs.append(SimpleTypeASMPLayer(in_features=hidden, out_features=hidden, use_x=use_x))
            self.batch_norm_list.append(BatchNorm(hidden, use_x))
            self.feature_extractors.append(GraphExtractor(in_features=hidden, out_features=hidden_final, use_x=use_x,
                                                          simplified=simplified))

        # Last layers
        self.simplified = simplified
        self.after_conv = nn.Linear(hidden_final, hidden_final)
        self.final_lin = nn.Linear(hidden_final, num_classes)

    def forward(self, data):
        """ data.x: (num_nodes, num_features)"""
        x, edge_index = data.x, data.edge_index
        batch_info = create_batch_info(data, self.edge_counter)

        # Create the context matrix
        if self.use_x:
            u = x
        else:
            u = data.x.new_zeros((data.num_nodes, batch_info['n_colors']))
            u.scatter_(1, data.coloring, 1)
            u = u[..., None]

        # Forward pass
        out = self.no_prop(u, batch_info)
        u = self.initial_lin(u)
        for i, (conv, bn, extractor) in enumerate(zip(self.convs, self.batch_norm_list, self.feature_extractors)):
            if self.use_batch_norm and i > 0:
                u = bn(u)
            u = conv(u, edge_index, batch_info)
            global_features = extractor.forward(u, batch_info)
            out += global_features / len(self.convs)

        # Two layer MLP with dropout and residual connections:
        if not self.simplified:
            out = torch.relu(self.after_conv(out)) + out
        out = F.dropout(out, p=self.dropout_prob, training=self.training)
        out = self.final_lin(out)
        return F.log_softmax(out, dim=-1)

    def reset_parameters(self):
        for layer in [self.no_prop, self.initial_lin, *self.convs, *self.batch_norm_list, *self.feature_extractors,
                      self.after_conv, self.final_lin]:
            layer.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__
