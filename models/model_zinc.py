import torch
import torch.nn.functional as F
import torch.nn as nn
from models.smp_layers import ZincSMPLayer
from models.utils.layers import GraphExtractor, EdgeCounter, BatchNorm
from models.utils.misc import create_batch_info, map_x_to_u


class SMPZinc(torch.nn.Module):
    def __init__(self, num_input_features: int, num_edge_features: int, num_classes: int, num_layers: int,
                 hidden: int, residual: bool, use_edge_features: bool, shared_extractor: bool,
                 hidden_final: int, use_batch_norm: bool, use_x: bool, map_x_to_u: bool,
                 num_towers: int, simplified: bool):
        """ num_input_features: number of node features
            num_edge_features: number of edge features
            num_classes: output dimension
            hidden: number of channels of the local contexts
            residual: use residual connexion after each SMP layer
            use_edge_features: if False, edge features are simply ignored
            shared extractor: share extractor among layers to reduce the number of parameters
            hidden_final: number of channels after extraction of graph features
            use_x: for ablation study, run a MPNN instead of SMP
            map_x_to_u: map the initial node features to the local context. If false, node features are ignored
            num_towers: inside each SMP layers, use towers to reduce the number of parameters
            simplified: if True, the feature extractor has less layers.
        """
        super().__init__()
        self.map_x_to_u, self.use_x = map_x_to_u, use_x
        self.use_batch_norm = use_batch_norm
        self.edge_counter = EdgeCounter()
        self.num_classes = num_classes
        self.residual = residual
        self.shared_extractor = shared_extractor

        self.no_prop = GraphExtractor(in_features=num_input_features, out_features=hidden_final, use_x=use_x)
        self.initial_lin = nn.Linear(num_input_features, hidden)

        self.convs = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        for i in range(0, num_layers):
            self.convs.append(ZincSMPLayer(in_features=hidden, num_towers=num_towers, out_features=hidden,
                                           edge_features=num_edge_features, use_x=use_x,
                                           use_edge_features=use_edge_features))
            self.batch_norm_list.append(BatchNorm(hidden, use_x) if i > 0 else None)

        # Feature extractors
        if shared_extractor:
            self.feature_extractor = GraphExtractor(in_features=hidden, out_features=hidden_final, use_x=use_x,
                                                    simplified=simplified)
        else:
            self.feature_extractors = torch.nn.ModuleList([])
            for i in range(0, num_layers):
                self.feature_extractors.append(GraphExtractor(in_features=hidden, out_features=hidden_final,
                                                              use_x=use_x, simplified=simplified))

        # Last layers
        self.after_conv = nn.Linear(hidden_final, hidden_final)
        self.final_lin = nn.Linear(hidden_final, num_classes)

    def forward(self, data):
        """ data.x: (num_nodes, num_node_features)"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Compute information about the batch
        batch_info = create_batch_info(data, self.edge_counter)

        # Create the context matrix
        if self.use_x:
            assert x is not None
            u = x
        elif self.map_x_to_u:
            u = map_x_to_u(data, batch_info)
        else:
            u = data.x.new_zeros((data.num_nodes, batch_info['n_colors']))
            u.scatter_(1, data.coloring, 1)
            u = u[..., None]

        # Forward pass
        out = self.no_prop(u, batch_info)
        u = self.initial_lin(u)
        for i in range(len(self.convs)):
            conv = self.convs[i]
            bn = self.batch_norm_list[i]
            extractor = self.feature_extractor if self.shared_extractor else self.feature_extractors[i]
            if self.use_batch_norm and i > 0:
                u = bn(u)
            u = conv(u, edge_index, edge_attr, batch_info) + (u if self.residual else 0)
            global_features = extractor.forward(u, batch_info)
            out += global_features / len(self.convs)

        out = self.final_lin(torch.relu(self.after_conv(out)) + out)
        assert out.shape[1] == 1
        return out[:, 0]

    def __repr__(self):
        return self.__class__.__name__
