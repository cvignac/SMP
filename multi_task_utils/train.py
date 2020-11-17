# This file was adapted from https://github.com/lukecavabarrett/pna

from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
from types import SimpleNamespace
import wandb
import numpy as np
import torch
import torch.optim as optim
import numpy.random as npr
from torch_geometric.data import DataLoader
from models.model_multi_task import SMP
from multi_task_utils.util import load_dataset, to_torch_geom, specific_loss_torch_geom

log_loss_tasks = ["log_shortest_path", "log_eccentricity", "log_laplacian",
                  "log_connected", "log_diameter", "log_radius"]


def build_arg_parser():
    """
    :return:    argparse.ArgumentParser() filled with the standard arguments for a training session.
                    Might need to be enhanced for some models.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='./data/multitask_dataset.pkl', help='Data path.')
    parser.add_argument('--gpu', type=int, help='Id of the GPU')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--only_nodes', action='store_true', default=False, help='Evaluate only nodes labels.')
    parser.add_argument('--only_graph', action='store_true', default=False, help='Evaluate only graph labels.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--patience', type=int, default=1000, help='Patience')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function to use.')
    parser.add_argument('--print_every', type=int, default=5, help='Print training results every')
    return parser


def execute_train(gnn_args, args):
    """
    :param gnn_args: the description of the model to be trained (expressed as arguments for GNN.__init__)
    :param args: the parameters of the training session
    """
    if not os.path.isdir('./saved_models'):
        os.mkdir('./saved_models')
    if args.name is not None:
        save_dir = f'./saved_models/{args.name}'
    else:
        save_dir = f'./saved_models/'
    if args.name is not None and not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    use_cuda = args.gpu is not None and torch.cuda.is_available() and not args.no_cuda
    if use_cuda:
        device = torch.device("cuda:" + str(args.gpu))
        torch.cuda.set_device(args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    else:
        device = "cpu"
    args.device = device
    args.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print('Using device:', device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # load data
    adj, features, node_labels, graph_labels = load_dataset(args.data, args.loss, args.only_nodes, args.only_graph,
                                                            print_baseline=True)
    print("Processing torch geometric data")
    graphs = to_torch_geom(adj, features, node_labels, graph_labels, device, args.debug)
    train_loaders = [DataLoader(given_size, args.batch_size, shuffle=True) for given_size in graphs['train']]
    batch_sizes = {'train': args.batch_size, 'val': 128, 'test': 256}
    val_loaders = [DataLoader(given_size, 128) for given_size in graphs['val']]
    test_loaders = [DataLoader(given_size, 256) for given_size in graphs['test']]
    print("Data loaders created")
    # model and optimizer
    gnn_args = SimpleNamespace(**gnn_args)

    gnn_args.num_input_features = features['train'][0].shape[2]
    gnn_args.nodes_out = 3
    gnn_args.graph_out = 3
    model = SMP(**vars(gnn_args)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=50,
                                                gamma=0.92)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params", pytorch_total_params)

    if args.load_from_epoch != -1:
        checkpoint = torch.load(os.path.join(save_dir, f'{args.load_from_epoch}.pkl'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0


    def train(epoch):
        """ Execute a single epoch of the training loop
        epoch (int): the number of the epoch being performed (0-indexed)."""
        t = time.time()

        # 1. Train
        nan_counts = 0
        model.train()
        total_train_loss_per_task = 0
        npr.shuffle(train_loaders)
        for i, loader in enumerate(train_loaders):
            for j, data in enumerate(loader):
                # Optimization
                optimizer.zero_grad()
                output = model(data.to(device))
                train_loss_per_task = specific_loss_torch_geom(output, (data.pos, data.y), data.batch, args.batch_size)
                loss_train = torch.mean(train_loss_per_task)
                if torch.isnan(loss_train):
                    print(f"Warning: loss was nan at epoch {epoch} and batch {i}{j}.")
                    nan_counts += 1
                    if nan_counts < 20:
                        continue
                    else:
                        raise ValueError(f"Too many NaNs. Stopping training at epoch {epoch}. Best epoch: {best_epoch}")
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                # Compute metrics
                total_train_loss_per_task += train_loss_per_task / len(loader)
        total_train_loss_per_task /= len(train_loaders)
        train_log_loss_per_task = torch.log10(total_train_loss_per_task).data.cpu().numpy()
        train_loss = torch.mean(total_train_loss_per_task).data.item()

        # validation epoch
        model.eval()
        val_loss_per_task = 0
        for loader in val_loaders:
            for i, data in enumerate(loader):
                if i > 0:
                    print("Warning: not all the batch was loaded at once. It will lead to incorrect results.")
                output = model(data.to(device))
                batch_loss_per_task = specific_loss_torch_geom(output, (data.pos, data.y), data.batch, batch_sizes['val'])
                val_loss_per_task += batch_loss_per_task.detach() / len(val_loaders)

        val_log_loss_per_task = torch.log10(val_loss_per_task).data.cpu().numpy()
        val_log_loss = torch.mean(val_loss_per_task).item()

        if epoch % args.print_every == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss.train: {:.4f}'.format(train_loss),
                  'log.loss.val: {:.4f}'.format(val_log_loss),
                  'time: {:.4f}s'.format(time.time() - t))
            print(f'train loss per task (log10 scale): {train_log_loss_per_task}')
            print(f'val loss per task (log10 scale): {val_log_loss_per_task}')
            sys.stdout.flush()
            if args.wandb:
                wandb_dict = {"Epoch": epoch, "Duration": time.time() - t, "Train loss": train_loss,
                              "Val log loss": val_log_loss}
                for loss, tr, val in zip(log_loss_tasks, train_log_loss_per_task, val_log_loss_per_task):
                    wandb_dict[loss + 'tr'] = tr
                    wandb_dict[loss + 'val'] = val
                wandb.log(wandb_dict)

        return val_log_loss

    def compute_test():
        """
        Evaluate the current model on all the sets of the dataset, printing results.
        This procedure is destructive on datasets.
        """
        model.eval()
        sets = list(features.keys())
        for dset, loaders in zip(sets, [train_loaders, val_loaders, test_loaders]):
            final_specific_loss = 0
            final_total_loss = 0
            for loader in loaders:
                loader_total_loss = 0
                loader_specific_loss = 0
                for data in loader:
                    output = model(data.to(device))
                    specific_loss = specific_loss_torch_geom(output, (data.pos, data.y),
                                                             data.batch, batch_sizes[dset]).detach()
                    loader_specific_loss += specific_loss
                    loader_total_loss += torch.mean(specific_loss)
                # Average the loss over each loader
                loader_specific_loss /= len(loader)
                loader_total_loss /= len(loader)
                # Average the loss over the different loaders
                final_specific_loss += loader_specific_loss / len(loaders)
                final_total_loss += loader_total_loss / len(loaders)
                del output, loader_specific_loss

            print("Test set results ", dset, ": loss= {:.4f}".format(final_total_loss))
            print(dset, ": ", final_specific_loss)
            print("Results in log scale", np.log10(final_specific_loss.detach().cpu()),
                  np.log10(final_total_loss.detach().cpu().numpy()))
        if args.wandb:
            wandb.run.summary["test results"] = np.log10(final_specific_loss.detach().cpu())
        # free unnecessary data


        final_specific_numpy = np.log10(final_specific_loss.detach().cpu())
        del final_total_loss, final_specific_loss
        torch.cuda.empty_cache()
        return final_specific_numpy

    sys.stdout.flush()
    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = -1

    sys.stdout.flush()

    while epoch < args.epochs:
        epoch += 1

        loss_values.append(train(epoch))
        scheduler.step()
        if epoch % 100 == 0:
            print("Results on the test set:")
            results_test = compute_test()
            print('Test set results', results_test)
            print(f"Saving checkpoint at epoch {epoch}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, f'{epoch}.pkl'))

        if loss_values[-1] < best:
            # save current model
            if loss_values[-1] < best:
                print(f"New best validation error at epoch {epoch}")
            else:
                print(f"Saving checkpoint at epoch {epoch}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(save_dir, f'{epoch}.pkl'))
            # remove previous model
            if best_epoch >= 0:
                f_name = os.path.join(save_dir, f'{best_epoch}.pkl')
                if os.path.isfile(f_name):
                    os.remove(f_name)
            # update training variables
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            print('Early stop at epoch {} (no improvement in last {} epochs)'.format(epoch + 1, bad_counter))
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch + 1))
    checkpoint = torch.load(os.path.join(save_dir, f'{best_epoch}.pkl'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Testing
    print("Results on the test set:")
    results_test = compute_test()
    print('Test set results', results_test)
