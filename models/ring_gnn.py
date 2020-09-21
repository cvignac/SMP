import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# import dgl
# from dgl.graph import DGLGraph
# from dgl.utils import Index

import pickle
import os

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


class RingGNN(nn.Module):
    def __init__(self, num_classes: int, num_layers: int, hidden: int, hidden_final: int, dropout_prob: float,
                 simplified: bool):
        super().__init__()
        self.layer_after_conv = not simplified
        self.dropout_prob = dropout_prob
        self.no_prop = FeatureExtractor(1, hidden_final)
        initial_conv = equi_2_to_2(1, hidden)
        self.convs = nn.ModuleList([initial_conv])
        self.bns = nn.ModuleList([])
        for i in range(1, num_layers):
            self.convs.append(equi_2_to_2(hidden, hidden))

        self.feature_extractors = torch.nn.ModuleList([])
        for i in range(num_layers):
            self.bns.append(nn.BatchNorm2d(hidden))
            self.feature_extractors.append(FeatureExtractor(hidden, hidden_final))
        if self.layer_after_conv:
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



class MLP(nn.Module):
    def __init__(self, feats):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(m, n) for m, n in zip(feats[:-1], feats[1:])])

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = layer(x)
            x = F.relu(x)

        return self.linears[-1](x)


# class Ring_GNN(nn.Module):
#     def __init__(self, nodeclasses, n_classes, avgnodenum=10, hidden=32, radius=2):
#         super(Ring_GNN, self).__init__()
#         self.depth = [th.LongTensor([nodeclasses]), th.LongTensor([64]), th.LongTensor([64])]
#         self.equi_modulelist = nn.ModuleList([equi_2_to_2(m, n, radius=radius, k2_init=0.5 / avgnodenum) for m, n in
#                                               zip(self.depth[:-1], self.depth[1:])])
#         self.prediction = MLP([th.sum(th.stack(self.depth)).item(), hidden, n_classes])
#
#     def forward(self, x):
#         x_list = [x]
#         for layer in self.equi_modulelist:
#             x = F.relu(layer(x))
#             x_list.append(x)
#
#         x_list = [th.sum(th.sum(x, dim=3), dim=2) for x in x_list]
#         x_list = th.cat(x_list, dim=1)
#         score = self.prediction(x_list)
#
#         return score


class equi_2_to_2(nn.Module):
    def __init__(self, input_depth, output_depth, normalization='inf', normalization_val=1.0, radius=2, k2_init=0.1):
        super(equi_2_to_2, self).__init__()
        basis_dimension = 15
        self.radius = radius
        # coeffs_values = lambda i, j, k: th.randn([i, j, k]) * th.sqrt(2. / (i + j).float())
        coeffs_values = lambda i, j, k: th.randn([i, j, k]) * np.sqrt(2. / float((i + j)))
        self.diag_bias_list = nn.ParameterList([])

        for i in range(radius):
            for j in range(i + 1):
                self.diag_bias_list.append(nn.Parameter(th.zeros(1, output_depth, 1, 1)))

        self.all_bias = nn.Parameter(th.zeros(1, output_depth, 1, 1))
        self.coeffs_list = nn.ParameterList([])

        for i in range(radius):
            for j in range(i + 1):
                self.coeffs_list.append(nn.Parameter(coeffs_values(input_depth, output_depth, basis_dimension)))

        self.switch = nn.ParameterList([nn.Parameter(th.FloatTensor([1])), nn.Parameter(th.FloatTensor([k2_init]))])
        self.output_depth = output_depth

        self.normalization = normalization
        self.normalization_val = normalization_val

    def forward(self, inputs):
        inputs = inputs.permute(0, 3, 1, 2)     # Convert to N x D x m x m
        m = inputs.size()[3]
        ops_out = ops_2_to_2(inputs, m, normalization=self.normalization)
        ops_out = th.stack(ops_out, dim=2)
        output_list = []

        for i in range(self.radius):
            for j in range(i + 1):
                output_i = th.einsum('dsb,ndbij->nsij', self.coeffs_list[i * (i + 1) // 2 + j], ops_out)
                mat_diag_bias = th.eye(inputs.size()[3]).to(inputs.device).unsqueeze(0).unsqueeze(0) * self.diag_bias_list[
                    i * (i + 1) // 2 + j]
                if j == 0:
                    output = output_i + mat_diag_bias
                else:
                    output = th.einsum('abcd,abde->abce', output_i, output)

            output_list.append(output)

        output = 0
        for i in range(self.radius):
            output += output_list[i] * self.switch[i]

        output = output + self.all_bias
        output = output.permute(0, 2, 3, 1)
        return output


def diag_offdiag_maxpool(input):
    max_diag = th.max(th.diagonal(input, dim1=2, dim2=3), dim=2)[0]

    max_val = th.max(max_diag)

    min_val = th.max(input * (-1.))
    val = th.abs(max_val + min_val)
    min_mat = th.diag_embed(th.diagonal(input[0][0]) * 0 + val).unsqueeze(0).unsqueeze(0)
    max_offdiag = th.max(th.max(input - min_mat, dim=2)[0], dim=2)[0]

    return th.cat([max_diag, max_offdiag], dim=1)


def ops_2_to_2(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
    # input: N x D x m x m
    diag_part = th.diagonal(inputs, dim1=2, dim2=3)  # N x D x m
    sum_diag_part = th.sum(diag_part, dim=2, keepdim=True)  # N x D x 1
    sum_of_rows = th.sum(inputs, dim=3)  # N x D x m
    sum_of_cols = th.sum(inputs, dim=2)  # N x D x m
    sum_all = th.sum(sum_of_rows, dim=2)  # N x D

    # op1 - (1234) - extract diag
    op1 = th.diag_embed(diag_part)  # N x D x m x m

    # op2 - (1234) + (12)(34) - place sum of diag on diag
    op2 = th.diag_embed(sum_diag_part.repeat(1, 1, dim))

    # op3 - (1234) + (123)(4) - place sum of row i on diag ii
    op3 = th.diag_embed(sum_of_rows)

    # op4 - (1234) + (124)(3) - place sum of col i on diag ii
    op4 = th.diag_embed(sum_of_cols)

    # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
    op5 = th.diag_embed(sum_all.unsqueeze(2).repeat(1, 1, dim))

    # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
    op6 = sum_of_cols.unsqueeze(3).repeat(1, 1, 1, dim)

    # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
    op7 = sum_of_rows.unsqueeze(3).repeat(1, 1, 1, dim)

    # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
    op8 = sum_of_cols.unsqueeze(2).repeat(1, 1, dim, 1)

    # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
    op9 = sum_of_rows.unsqueeze(2).repeat(1, 1, dim, 1)

    # op10 - (1234) + (14)(23) - identity
    op10 = inputs

    # op11 - (1234) + (13)(24) - transpose
    op11 = th.transpose(inputs, -2, -1)

    # op12 - (1234) + (234)(1) - place ii element in row i
    op12 = diag_part.unsqueeze(3).repeat(1, 1, 1, dim)

    # op13 - (1234) + (134)(2) - place ii element in col i
    op13 = diag_part.unsqueeze(2).repeat(1, 1, dim, 1)

    # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
    op14 = sum_diag_part.unsqueeze(3).repeat(1, 1, dim, dim)

    # op15 - sum of all ops - place sum of all entries in all entries
    op15 = sum_all.unsqueeze(2).unsqueeze(3).repeat(1, 1, dim, dim)

    # A_2 = th.einsum('abcd,abde->abce', inputs, inputs)
    # A_4 = th.einsum('abcd,abde->abce', A_2, A_2)
    # op16 = th.where(A_4>1, th.ones(A_4.size()), A_4)

    if normalization is not None:
        float_dim = float(dim)
        if normalization is 'inf':
            op2 = th.div(op2, float_dim)
            op3 = th.div(op3, float_dim)
            op4 = th.div(op4, float_dim)
            op5 = th.div(op5, float_dim ** 2)
            op6 = th.div(op6, float_dim)
            op7 = th.div(op7, float_dim)
            op8 = th.div(op8, float_dim)
            op9 = th.div(op9, float_dim)
            op14 = th.div(op14, float_dim)
            op15 = th.div(op15, float_dim ** 2)

    # return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16]
    '''
    l = [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]
    for i, ls in enumerate(l):
        print(i+1)
        print(th.sum(ls))
    print("$%^&*(*&^%$#$%^&*(*&^%$%^&*(*&^%$%^&*(")
    '''
    return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]

#
# def input_single():
#     # np.random.seed(18)
#     iso = False
#     adj_list = pickle.load(open(os.path.join('Synthetic_Data', 'graphs_Kary_Deterministic_Graphs.pkl'), 'rb'))
#     y = th.load(os.path.join('Synthetic_Data', 'y_Kary_Deterministic_Graphs.pt'))
#
#     while True:
#         selected_index = np.random.randint(low=0, high=150, size=2)
#         if iso and y[selected_index[0]] == y[selected_index[1]]:
#             break
#         if iso == False and y[selected_index[0]] != y[selected_index[1]]:
#             break
#
#     adj_list = [adj_list[i] for i in selected_index]
#     print(selected_index)
#     _, G = extract_deg_adj(convert_to_graph([adj_list[0]]))
#     _, G_prime = extract_deg_adj(convert_to_graph([adj_list[1]]))
#     G = G[0].to_dense()
#     G = G.unsqueeze(0).unsqueeze(0)
#     G_prime = G_prime[0].to_dense()
#     G_prime = G_prime.unsqueeze(0).unsqueeze(0)
#     return G, G_prime

#
# def convert_to_graph(coo_list):
#     graph_list = []
#     for coo in coo_list:
#         g = dgl.DGLGraph()
#         g.from_scipy_sparse_matrix(coo)
#         graph_list.append(g)
#
#     return graph_list
#
#
# def extract_deg_adj(graph_list):
#     in_degrees = lambda g: g.in_degrees(
#         Index(np.arange(0, g.number_of_nodes()))).unsqueeze(1).float()
#     degs = [in_degrees(g) for g in graph_list]
#     adjs = [g.adjacency_matrix() for g in graph_list]
#     return degs, adjs
#
#
# def test():
#     model = Ring_GNN()
#     dim = 20
#     test_input = th.ones(th.Size([5, 4, 20, 20]))
#     print(model(test_input))
#     print('ok')
#
#
# def main():
#     model = Ring_GNN()
#     input1, input2 = input_single()
#     print(model(input1))
#     print(model(input1) - model(input2))
#
#
# if __name__ == '__main__':
#     main()
