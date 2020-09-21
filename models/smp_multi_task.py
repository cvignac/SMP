import torch
import torch.nn as nn
from models.utils.layers import EdgeCounter, NodeExtractor, BatchNorm
from models.smp_layers import FastSMPLayer, SMPLayer, SimplifiedFastSMPLayer
from torch_geometric.nn import Set2Set
from models.utils.misc import create_batch_info


class SMP(torch.nn.Module):
    def __init__(self, num_input_features: int, nodes_out: int, graph_out: int,
                 num_layers: int, num_towers: int, hidden_u: int, out_u: int, hidden_gru: int,
                 layer_type: str):
        super().__init__()
        num_input_u = 1 + num_input_features

        self.edge_counter = EdgeCounter()
        self.initial_lin_u = nn.Linear(num_input_u, hidden_u)

        self.extractor = NodeExtractor(hidden_u, out_u)

        layer_type_dict = {'SMP': SMPLayer, 'FastSMP': FastSMPLayer, 'SimplifiedFastSMP': SimplifiedFastSMPLayer}
        conv_layer = layer_type_dict[layer_type]

        self.gru = nn.GRU(out_u, hidden_gru)
        self.convs = nn.ModuleList([])
        self.batch_norm_u = nn.ModuleList([])
        for i in range(0, num_layers):
            self.batch_norm_u.append(BatchNorm(hidden_u, use_x=False))
            conv = conv_layer(in_features=hidden_u, out_features=hidden_u, num_towers=num_towers, use_x=False)
            self.convs.append(conv)

        # Process the extracted node features
        max_n = 19
        self.set2set = Set2Set(hidden_gru, max_n)

        self.final_node = nn.Sequential(nn.Linear(hidden_gru, hidden_gru), nn.LeakyReLU(),
                                        nn.Linear(hidden_gru, hidden_gru), nn.LeakyReLU(),
                                        nn.Linear(hidden_gru, nodes_out))

        self.final_graph = nn.Sequential(nn.Linear(2 * hidden_gru, hidden_gru), nn.ReLU(),
                                         nn.BatchNorm1d(hidden_gru),
                                         nn.Linear(hidden_gru, hidden_gru), nn.LeakyReLU(),
                                         nn.BatchNorm1d(hidden_gru),
                                         nn.Linear(hidden_gru, graph_out))

    def forward(self, data):
        """ data.x: (num_nodes, num_features)"""
        x, edge_index, batch, batch_size = data.x, data.edge_index, data.batch, data.num_graphs
        batch_info = create_batch_info(data, self.edge_counter)

        # Create the context matrix
        u = data.x.new_zeros((data.num_nodes, batch_info['n_colors']))
        u.scatter_(1, data.coloring, 1)
        u = u[..., None]

        # Map x to u
        shortest_path_ids = x[:, 0]
        lap_feat = x[:, 1]
        u_shortest_path = torch.zeros_like(u)
        u_lap_feat = torch.zeros_like(u)
        non_zero = shortest_path_ids.nonzero(as_tuple=False)[:, 0]
        nonzero_batch = batch_info['batch'][non_zero]
        nonzero_color = batch_info['coloring'][non_zero][:, 0]
        for b, c in zip(nonzero_batch, nonzero_color):
            u_shortest_path[batch == b, c] = 1

        for i, feat in enumerate(lap_feat):
            u_lap_feat[i, batch_info['coloring'][i]] = feat

        u = torch.cat((u, u_shortest_path, u_lap_feat), dim=2)

        # Forward pass
        u = self.initial_lin_u(u)
        hidden_state = None
        for i, (conv, bn_u) in enumerate(zip(self.convs,  self.batch_norm_u)):
            if i > 0:
                u = bn_u(u)
            u = conv(u, edge_index, batch_info)
            extracted = self.extractor(x, u, batch_info)[None, :, :]
            hidden_state = self.gru(extracted, hidden_state)[1]

        # Compute the final representation
        out = hidden_state[0, :, :]
        nodes_out = self.final_node(out)
        after_set2set = self.set2set(out, batch_info['batch'])
        graph_out = self.final_graph(after_set2set)

        return nodes_out, graph_out

    def __repr__(self):
        return self.__class__.__name__
