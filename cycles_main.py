#!/usr/bin/env python
# coding: utf-8

# ## Example k-cycle classification


import os
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.transforms import OneHotDegree
import argparse
import numpy as np
import time
import yaml
from models.smp_cycles import SMP
from models.gin import GIN
from datasets_generation.build_cycles import FourCyclesDataset
from models.utils.transforms import EyeTransform, RandomId, DenseAdjMatrix
from models import ppgn
from models.ring_gnn import RingGNN
from easydict import EasyDict as edict

# Change the following to point to the the folder where the datasets are stored
if os.path.isdir('/datasets2/'):
    rootdir = '/datasets2/CYCLE_DETECTION/'
else:
    rootdir = './data/datasets_kcycle_nsamples=10000/'
yaml_file = './config_cycles.yaml'
# yaml_file = './benchmark/kernel/config4cycles.yaml'
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--k', type=int, default=4,
                    help="Length of the cycles to detect")
parser.add_argument('--n', type=int, help='Average number of nodes in the graphs')
parser.add_argument('--save-model', action='store_true',
                    help='Save the model once training is done')
parser.add_argument('--wandb', action='store_true',
                    help="Use weights and biases library")
parser.add_argument('--gpu', type=int, help='Id of gpu device. By default use cpu')
parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate")
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--weight-decay', type=float, default=1e-6)
parser.add_argument('--clip', type=float, default=10, help="Gradient clipping")
parser.add_argument('--name', type=str, help="Name for weights and biases")
parser.add_argument('--proportion', type=float, default=1.0,
                    help='Proportion of the training data that is kept')
parser.add_argument('--generalization', action='store_true',
                    help='Evaluate out of distribution accuracy')
args = parser.parse_args()

# Log parameters
test_every_epoch = 5
print_every_epoch = 1
log_interval = 20

# Store maximum number of nodes for each pair (k, n)
# Used by provably powerful graph networks
max_num_nodes = {4: {12: 12, 20: 20, 28: 28, 36: 36},
                 6: {20: 25, 31: 38, 42: 52, 56: 65},
                 8: {28: 38, 50: 56, 66: 76, 72: 90}}
# Store the maximum degree for the one-hot encoding
max_degree = {4: {12: 4, 20: 6, 28: 7, 36: 7},
              6: {20: 4, 31: 6, 42: 8, 56: 7},
              8: {28: 4, 50: 6, 66: 7, 72: 8}}
# Store the values of n to use for generalization experiments
n_gener = {4: {'train': 20, 'val': 28, 'test': 36},
           6: {'train': 31, 'val': 42, 'test': 56},
           8: {'train': 50, 'val': 66, 'test': 72}}

if args.name:
    args.wandb = True

# Handle the device
use_cuda = args.gpu is not None and torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] =     str(args.gpu)
else:
    device = "cpu"
args.device = device
args.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print('Device used:', device)

# Load the config file of the model
with open(yaml_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = edict(config)
    print(config)

model_name = config['model_name']
config.pop('model_name')
if model_name == 'SMP':
    model_name = config['layer_type']



if args.name is None:
    if model_name != 'GIN':
        args.name = model_name
    else:
        if config.relational_pooling > 0:
            args.name = 'RP'
        if config.one_hot:
            args.name = 'OneHotDeg'
        elif config.identifiers:
            args.name = 'OneHotNod'
        elif config.random:
            args.name = 'Random'
        else:
            args.name = 'GIN'
    args.name = args.name + '_' + str(args.k)
    if args.n is not None:
        args.name = args.name + '_' + str(args.n)

# Create a folder for the saved models
if not os.path.isdir('./saved_models/' + args.name) and args.generalization:
    os.mkdir('./saved_models/' + args.name)

if args.wandb:
    import wandb
    wandb.init(project="smp", config=config, name=args.name)
    wandb.config.update(args)

if args.n is None:
    args.n = n_gener[args.k]['train']



def train(epoch):
    """ Train for one epoch. """
    model.train()
    lr_scheduler(args.lr, epoch, optimizer)
    loss_all = 0
    if not config.relational_pooling:
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
        return loss_all / len(train_loader.dataset)
    else:
        for batch_idx, data in enumerate(train_loader):
            for repetition in range(args.relational_pooling):
                for i in range(args.batch_size):
                    n_nodes = int(torch.sum(data.batch == i).item())
                    p = torch.randperm(n_nodes)
                    data.x[data.batch == i, :n_nodes] = data.x[data.batch == i, :n_nodes][p, :][:, p]
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, data.y)
                loss.backward()
                loss_all += loss.item() * data.num_graphs
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
        return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


def lr_scheduler(lr, epoch, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * (0.995 ** (epoch / 5))


# Define the transform to use in the dataset
if model_name == 'GIN':
    if config.one_hot:
        max_degree = max_degree[args.k][args.n]
        transform = OneHotDegree(max_degree, cat=False)
        config.num_input_features = max_degree + 1
    elif config.identifiers:
        transform = EyeTransform(max_num_nodes[args.k][args.n])
        config.num_input_features = max_num_nodes[args.k][args.n]
    elif config.random:
        transform = RandomId()
        config.num_input_features = 1
else:
    transform = None
    config.num_input_features = 1


start = time.time()

if config.num_layers == -1:
    config.num_layers = args.k

if 'SMP' in model_name:
    config.use_batch_norm = args.k > 6 or args.n > 30
    model = SMP(config.num_input_features, config.num_classes, config.num_layers, config.hidden, config.layer_type,
                 config.hidden_final, config.dropout_prob, config.use_batch_norm, config.use_x, config.map_x_to_u,
                 config.num_towers, config.simplified).to(device)
elif model_name == 'PPGN':
    transform = DenseAdjMatrix(max_num_nodes[args.k][args.n])
    model = ppgn.Powerful(config.num_classes, config.num_layers, config.hidden,
                          config.hidden_final, config.dropout_prob, config.simplified)
elif model_name == 'GIN':
    config.use_batch_norm = args.k > 6 or args.n > 50
    model = GIN(config.num_input_features, config.num_classes, config.num_layers,
                config.hidden, config.hidden_final, config.dropout_prob, config.use_batch_norm)
elif model_name == 'RING_GNN':
    transform = DenseAdjMatrix(max_num_nodes[args.k][args.n])
    model = RingGNN(config.num_classes, config.num_layers, config.hidden, config.hidden_final, config.dropout_prob,
                    config.layer_after_conv)

model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay=args.weight_decay)
# Load the data
print("Transform used:", transform)

batch_size = args.batch_size
if args.generalization:
    train_data = FourCyclesDataset(args.k, n_gener[args.k]['train'], rootdir, train=True, transform=transform)
    test_data = FourCyclesDataset(args.k, n_gener[args.k]['train'], rootdir, train=False, transform=transform)
    gener_data_val = FourCyclesDataset(args.k, n_gener[args.k]['val'], rootdir, train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)
    gener_val_loader = DataLoader(gener_data_val, batch_size, shuffle=False)

else:
    train_data = FourCyclesDataset(args.k, args.n, rootdir, proportion=args.proportion, train=True, transform=transform)
    test_data = FourCyclesDataset(args.k, args.n, rootdir, proportion=args.proportion, train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)



print("Starting to train")
best_epoch = -1
best_generalization_acc = 0
for epoch in range(args.epochs):
    epoch_start = time.time()
    tr_loss = train(epoch)
    if epoch % print_every_epoch == 0:
        acc_train = test(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        duration = time.time() - epoch_start
        print(f'Time:{duration:2.2f} | {epoch:5d} | Loss: {tr_loss:2.5f} | Train Acc: {acc_train:2.5f} | LR: {current_lr:.6f}')
        if epoch % test_every_epoch == 0:
            acc_test = test(test_loader)
            print(f'Test accuracy: {acc_test:2.5f}')
            if args.generalization:
                acc_generalization = test(gener_val_loader)
                if args.wandb:
                    wandb.log({"Epoch": epoch, "Duration": duration, "Train loss": tr_loss, "train accuracy": acc_train,
                               "Test acc": acc_test, 'Gene eval': acc_generalization})
                if acc_generalization > best_generalization_acc and acc_test > 0.9:
                    print(f"New best generalization error + accuracy > 90% at epoch {epoch}")
                    # Remove existing models
                    folder = f'./saved_models/{args.name}/'
                    files_in_folder = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
                    for file in files_in_folder:
                        try:
                            os.remove(file)
                        except:
                            print("Could not remove file", file)
                    # Save new model
                    torch.save(model, f'./saved_models/{args.name}/epoch{epoch}.pkl')
                    print(f"Model saved at epoch {epoch}.")
                    best_epoch = epoch
            else:
                if args.wandb:
                    wandb.log({"Epoch": epoch, "Duration": duration, "Train loss": tr_loss, "train accuracy": acc_train,
                               "Test acc": acc_test})
        else:
            if args.wandb:
                wandb.log({"Epoch": epoch, "Duration": duration, "Train loss": tr_loss, "train accuracy": acc_train})

cur_lr = optimizer.param_groups[0]["lr"]
print(f'{epoch:2.5f} | Loss: {tr_loss:2.5f} | Train Acc: {acc_train:2.5f} | LR: {cur_lr:.6f} | Test Acc: {acc_test:2.5f}')
print(f'Elapsed time: {(time.time() - start) / 60:.1f} minutes')
print('done!')

final_acc = test(test_loader)
print(f"Final accuracy: {final_acc}")
print("Done.")

if args.generalization:
    new_n = n_gener[args.k]['test']
    gener_data_test = FourCyclesDataset(args.k, new_n, rootdir, train=False, transform=transform)
    gener_test_loader = DataLoader(gener_data_test, batch_size, shuffle=False)
    model = model.load_state_dict(torch.load(f"./saved_models/{args.name}/epoch{best_epoch}.pkl", map_location=device))
    acc_test_generalization = test(gener_test_loader)
    print(f"Generalization accuracy on {args.k} cycles with {new_n} nodes", acc_test)
    if args.wandb:
        wandb.run.summary['test_generalization'] = acc_test
