import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
    def __init__(self, num_classes: int, num_layers: int, hidden: int, hidden_final: int, dropout_prob: float,
                 layer_after_conv: bool = False):
        super().__init__()
        layers_per_conv = 1
        self.layer_after_conv = layer_after_conv
        self.dropout_prob = dropout_prob
        self.no_prop = FeatureExtractor(1, hidden_final)
        initial_conv = PowerfulLayer(1, hidden, layers_per_conv)
        self.convs = nn.ModuleList([initial_conv])
        self.bns = nn.ModuleList([])
        for i in range(1, num_layers):
            self.convs.append(PowerfulLayer(hidden, hidden, layers_per_conv))

        self.feature_extractors = torch.nn.ModuleList([])
        for i in range(num_layers):
            self.bns.append(nn.BatchNorm2d(hidden))
            self.feature_extractors.append(FeatureExtractor(hidden, hidden_final))
        self.layer_after_conv = layer_after_conv
        if layer_after_conv:
            self.after_conv = nn.Linear(hidden_final, hidden_final)
        self.final_lin = nn.Linear(hidden_final, num_classes)

    def forward(self, data):
        u = data.A[..., None]           # batch, N, N, 1
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
