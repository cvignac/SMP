import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor as Tensor
from models.smp_layers import SimplifiedFastSMPLayer, FastSMPLayer, SMPLayer, ZincSMPLayer, FilmZincSMPLayer
from models.utils.layers import GraphExtractor, EdgeCounter, BatchNorm
from models.utils.misc import create_batch_info, map_x_to_u
from torch_geometric.utils import to_dense_adj
from models.smp_cycles import SMP
import matplotlib.pyplot as plt


class SMPPPGN(torch.nn.Module):
    def __init__(self, num_input_features: int, num_classes: int, num_layers: int, hidden: int, layer_type: str,
                 hidden_final: int, dropout_prob: float, use_batch_norm: bool, use_x: bool, map_x_to_u: bool,
                 num_towers: int, simplified: bool):
        super().__init__()
        self.map_x_to_u, self.use_x = map_x_to_u, use_x
        self.dropout_prob = dropout_prob
        self.use_batch_norm = use_batch_norm
        self.edge_counter = EdgeCounter()
        self.num_classes = num_classes

        self.no_prop = GraphExtractor(in_features=num_input_features, out_features=hidden_final, use_x=use_x)
        self.initial_lin = nn.Linear(num_input_features, hidden)

        layer_type_dict = {'SMP': SMPLayer, 'FastSMP': FastSMPLayer, 'SimplifiedFastSMP': SimplifiedFastSMPLayer}
        conv_layer = layer_type_dict[layer_type]

        self.convs = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        self.feature_extractors = torch.nn.ModuleList([])
        for i in range(0, num_layers):
            self.convs.append(conv_layer(in_features=hidden, num_towers=num_towers, out_features=hidden, use_x=use_x))
            self.batch_norm_list.append(BatchNorm(hidden, use_x))
            self.feature_extractors.append(Powerful(hidden, hidden_final, 2, hidden, hidden_final, dropout_prob=0, simplified=False))

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
        for i, (conv, bn, extractor) in enumerate(zip(self.convs, self.batch_norm_list, self.feature_extractors)):
            if self.use_batch_norm and i > 0:
                u = bn(u)
            u = conv(u, edge_index, batch_info)
            global_features = extractor.forward(u, edge_index, batch_info)
            out += global_features / len(self.convs)

        # Two layer MLP with dropout and residual connections:
        if not self.simplified:
            out = torch.relu(self.after_conv(out)) + out
        out = F.dropout(out, p=self.dropout_prob, training=self.training)
        out = self.final_lin(out)
        if self.num_classes > 1:
            # Classification
            return F.log_softmax(out, dim=-1)
        else:
            assert out.shape[1] == 1
            return out[:, 0]

    def reset_parameters(self):
        for layer in [self.no_prop, self.initial_lin, *self.convs, *self.batch_norm_list, *self.feature_extractors,
                      self.after_conv, self.final_lin]:
            layer.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__


class InvariantMaxLayer(nn.Module):
    def forward(self, x: Tensor):
        """ x: (batch_size, n_nodes, n_nodes, channels)"""
        bs, n, channels = x.shape[0], x.shape[1], x.shape[3]
        diag = torch.diagonal(x, dim1=1, dim2=2).contiguous()        # batch, Channels, n_nodes
        # max_diag = diag.max(dim=2)[0]                                # Batch, channels
        max_diag = diag.sum(dim=2)
        mask = ~ torch.eye(n=x.shape[1], dtype=torch.bool, device=x.device)[None, :, :, None].expand(x.shape)
        x_off_diag = x[mask].reshape(bs, n, n - 1, channels)
        # max_off_diag = x_off_diag.max(dim=1)[0].max(dim=1)[0]
        max_off_diag = x_off_diag.sum(dim=1).sum(dim=1)
        out = torch.cat((max_diag, max_off_diag), dim=1)
        return out


class UnitMLP(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_feat, out_feat, (1, 1)))
        for i in range(1, num_layers):
            self.layers.append(nn.Conv2d(out_feat, out_feat, (1, 1)))

    def forward(self, x: Tensor):
        """ x: batch x N x N x channels"""
        # Convert for conv2d
        x = x.permute(0, 3, 1, 2).contiguous()                   # channels, N, N
        for layer in self.layers[:-1]:
            x = F.relu(layer.forward(x))
        x = self.layers[-1].forward(x)
        x = x.permute(0, 2, 3, 1)           # batch_size, N, N, channels
        return x


class PowerfulLayer(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, num_layers: int):
        super().__init__()
        a = in_feat
        b = out_feat
        self.m1 = UnitMLP(a, b, num_layers)
        self.m2 = UnitMLP(a, b, num_layers)
        self.m4 = nn.Linear(a + b, b, bias=True)

    def forward(self, x):
        """ x: batch x N x N x in_feat"""
        out1 = self.m1.forward(x).permute(0, 3, 1, 2)                 # batch, out_feat, N, N
        out2 = self.m2.forward(x).permute(0, 3, 1, 2)                 # batch, out_feat, N, N
        out3 = x
        mult = out1 @ out2                                         # batch, out_feat, N, N
        out = torch.cat((mult.permute(0, 2, 3, 1), out3), dim=3)      # batch, N, N, out_feat
        suffix = self.m4.forward(out)
        return suffix


class FeatureExtractor(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin1 = nn.Linear(in_features, out_features, bias=True)
        self.lin2 = nn.Linear(in_features, out_features, bias=False)
        self.lin3 = torch.nn.Linear(out_features, out_features, bias=False)

    def forward(self, u):
        """ u: (batch_size, num_nodes, num_nodes, in_features)
            output: (batch_size, out_features). """
        n = u.shape[1]
        diag = u.diagonal(dim1=1, dim2=2)       # batch_size, channels, num_nodes
        trace = torch.sum(diag, dim=2)
        out1 = self.lin1.forward(trace / n)

        s = (torch.sum(u, dim=[1, 2]) - trace) / (n * (n-1))
        out2 = self.lin2.forward(s)  # bs, out_feat
        out = out1 + out2
        out = out + self.lin3.forward(F.relu(out))
        return out


class Powerful(nn.Module):
    def __init__(self, in_feat: int, num_classes: int, num_layers: int, hidden: int, hidden_final: int, dropout_prob: float,
                 simplified: bool):
        super().__init__()
        layers_per_conv = 1
        self.layer_after_conv = not simplified
        self.dropout_prob = dropout_prob
        self.no_prop = FeatureExtractor(in_feat, hidden_final)
        initial_conv = PowerfulLayer(in_feat, hidden, layers_per_conv)
        self.convs = nn.ModuleList([initial_conv])
        self.bns = nn.ModuleList([])
        for i in range(1, num_layers):
            self.convs.append(PowerfulLayer(hidden, hidden, layers_per_conv))

        self.feature_extractors = torch.nn.ModuleList([])
        for i in range(num_layers):
            self.bns.append(nn.BatchNorm2d(hidden))
            self.feature_extractors.append(FeatureExtractor(hidden, hidden_final))
        if self.layer_after_conv:
            self.after_conv = nn.Linear(hidden_final, hidden_final)
        self.final_lin = nn.Linear(hidden_final, num_classes)

    def forward(self, u, edge_index, batch_info):
        u = u.unsqueeze(0)
        out = self.no_prop.forward(u)
        for conv, extractor, bn in zip(self.convs, self.feature_extractors, self.bns):
            u = conv(u)
            u = bn(u.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            out = out + extractor.forward(u)
        out = F.relu(out) / len(self.convs)
        if self.layer_after_conv:
            out = out + F.relu(self.after_conv(out))
        out = F.dropout(out, p=self.dropout_prob, training=self.training)
        out = self.final_lin(out)
        return F.log_softmax(out, dim=-1)


x = torch.ones(16, dtype=torch.float32)

ham = torch.from_numpy(np.loadtxt('./tests/hamming2.4.am.csv', delimiter=','))
ham = Data(x=x, edge_index=dense_to_sparse(ham)[0])
batch_ham = Batch.from_data_list([ham])

perm = np.random.permutation(16)
adj = np.loadtxt('./tests/hamming2.4.am.csv', delimiter=',')[perm, :][:, perm]
adj = torch.from_numpy(adj)
ham2 = Data(x=x, edge_index=dense_to_sparse(adj)[0])
batch_ham2 = Batch.from_data_list([ham2])

shri = torch.from_numpy(np.loadtxt('./tests/shrikhande.am.csv', delimiter=','))
shri = Data(x=x, edge_index=dense_to_sparse(shri)[0])
batch_shri = Batch.from_data_list([shri])

diffs1 = []
diffs2 = []
k = 1000
for i in range(k):
    model = SMPPPGN(num_input_features=1,  num_classes=1, num_layers=4, hidden=4, layer_type='SMP', hidden_final=4,
                dropout_prob=0, use_batch_norm=False, use_x=False, map_x_to_u=False, num_towers=1, simplified=False)


    out1 = model(batch_shri).item()
    out2 = model(batch_ham).item()
    out3 = model(batch_ham2).item()
    print(out1, out3 - out2, out2 - out1)
    diffs1.append(np.log10(out3 - out2 + 1e-10))
    diffs2.append(np.log10(out2 - out1 + 1e-10))

    if abs(out2 - out1) > 1e-4:
        print("Warning: it seems to be different")
        break
    if abs(out3 - out2) > 1e-4:
        print("The same graph used twice gave substancially different results")
        break

diffs1 = np.array(diffs1)
diffs2 = np.array(diffs2)
plt.hist(diffs1[diffs1 > -9.5], color='blue', bins=int(2*np.sqrt(k)), alpha=0.5, log=True)
plt.hist(diffs2[diffs2 > -9.5], color='red', bins=int(2*np.sqrt(k)), alpha=0.5, log=True)
plt.show()