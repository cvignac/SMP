import torch


def create_batch_info(data, edge_counter):
    """ Compute some information about the batch that will be used by SMP."""
    x, edge_index, batch, batch_size = data.x, data.edge_index, data.batch, data.num_graphs

    # Compute some information about the batch
    # Count the number of nodes in each graph
    unique, n_per_graph = torch.unique(data.batch, return_counts=True)
    n_batch = torch.zeros_like(batch, dtype=torch.float)

    for value, n in zip(unique, n_per_graph):
        n_batch[batch == value] = n.float()

    # Count the average number of edges per graph
    dummy = x.new_ones((data.num_nodes, 1))
    average_edges = edge_counter(dummy, edge_index, batch, batch_size)

    # Create the coloring if it does not exist yet
    if not hasattr(data, 'coloring'):
        data.coloring = data.x.new_zeros(data.num_nodes, dtype=torch.long)
        for i in range(data.num_graphs):
            data.coloring[data.batch == i] = torch.arange(n_per_graph[i], device=data.x.device)
        data.coloring = data.coloring[:, None]
    n_colors = torch.max(data.coloring) + 1  # Indexing starts at 0

    mask = torch.zeros(data.num_nodes, n_colors, dtype=torch.bool, device=x.device)
    for value, n in zip(unique, n_per_graph):
        mask[batch == value, :n] = True

    # Aggregate into a dict
    batch_info = {'num_nodes': data.num_nodes,
                  'num_graphs': data.num_graphs,
                  'batch': data.batch,
                  'n_per_graph': n_per_graph,
                  'n_batch': n_batch[:, None, None].float(),
                  'average_edges': average_edges[:, :, None],
                  'coloring': data.coloring,
                  'n_colors': n_colors,
                  'mask': mask}
    return batch_info