import os
import torch
import pickle
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import networkx as nx
import numpy.random as npr


if os.path.isdir('/datasets2/'):
    rootdir = '/datasets2/CYCLE_DETECTION/'
else:
    rootdir = './data/datasets_kcycle_nsamples=10000/'


def build_dataset():
    """ Given pickle files, split the dataset into one per value of n
    Run once before running the experiments. """
    n_samples = 10000
    for k in [4, 6, 8]:
        with open(os.path.join(rootdir, 'datasets_kcycle_k={}_nsamples=10000.pickle'.format(k)), 'rb') as f:
            datasets_params, datasets = pickle.load(f)
            # Split by graph size
            for params, dataset in zip(datasets_params, datasets):
                n = params['n']
                train, test = dataset[:n_samples], dataset[n_samples:]
                torch.save(train, rootdir + f'{k}cycles_n{n}_{n_samples}samples_train.pt')
                torch.save(test, rootdir + f'/{k}cycles_n{n}_{n_samples}samples_test.pt')
                # torch.save(test, '{}cycles_n{}_{}samples_test.pt'.format(k, n, n_samples))


class FourCyclesDataset(InMemoryDataset):
    def __init__(self, k, n, root, train, proportion=1.0, n_samples=10000, transform=None, pre_transform=None):
        self.train = train
        self.k, self.n, self.n_samples = k, n, n_samples
        self.root = root
        self.s = 'train' if train else 'test'
        self.proportion = proportion
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['{}cycles_n{}_{}samples_{}.pt'.format(self.k, self.n, self.n_samples, self.s)]

    @property
    def processed_file_names(self):
        if self.transform is None:
            st = 'no-transf'
        else:
            st = str(self.transform.__class__.__name__)
        return [f'processed_{self.k}cycles_n{self.n}_{self.n_samples}samples_{self.s}_{st}_{self.proportion}.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        dataset = torch.load(os.path.join(self.root, f'{self.k}cycles_n{self.n}_{self.n_samples}samples_{self.s}.pt'))

        data_list = []
        for sample in dataset:
            graph, y, label = sample
            edge_list = nx.to_edgelist(graph)
            edges = [np.array([edge[0], edge[1]]) for edge in edge_list]
            edges2 = [np.array([edge[1], edge[0]]) for edge in edge_list]

            edge_index = torch.tensor(np.array(edges + edges2).T, dtype=torch.long)

            x = torch.ones(graph.number_of_nodes(), 1, dtype=torch.float)
            y = torch.tensor([1], dtype=torch.long) if label == 'has-kcycle' else torch.tensor([0], dtype=torch.long)

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=None, y=y))
        # Subsample the data
        if self.train:
            all_data = len(data_list)
            to_select = int(all_data * self.proportion)
            print(to_select, "samples were selected")
            data_list = data_list[:to_select]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    build_dataset()