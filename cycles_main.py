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
parser.add_argument('--k', type=int, default=6,
                    help="Length of the cycles to detect")
parser.add_argument('--n', type=int, default=56,
                    help='Average number of nodes in the graphs')
parser.add_argument('--save-model', action='store_true',
                    help='Save the model once training is done')
parser.add_argument('--wandb', action='store_true',
                    help="Use weights and biases library")
parser.add_argument('--gpu', type=int, help='Id of gpu device. By default use cpu')
parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate")
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--weight-decay', type=float, default=0.0001)
parser.add_argument('--clip', type=float, default=10, help="Gradient clipping")
parser.add_argument('--one-hot', action='store_true', default=False,
                    help='Use a one-hot encoding of the degree as node features')
parser.add_argument('--identifiers', action='store_true', default=False,
                    help='Use a one hot encoding of the nodes as node features.')
parser.add_argument('--random', action='store_true', help="Use random identifiers as node features.")
parser.add_argument('--name', type=str, help="Name for weights and biases")
parser.add_argument('--proportion', type=float, default=1.0,
                    help='Proportion of the training data that is kept')
args = parser.parse_args()

# Log parameters
test_every_epoch = 5
print_every_epoch = 1
log_interval = 20

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
    model_config = yaml.load(f, Loader=yaml.FullLoader)
    print(model_config)

model_name = model_config['model_name']
model_config.pop('model_name')
print("Model name:", model_name)

# Create a name for weights and biases
if args.wandb:
    import wandb
    if args.name is None:
        if args.random:
            args.name = 'random' + f'_{args.k}_{args.n}'
        else:
            args.name = model_name + f'_{args.k}_{args.n}'
    wandb.init(project="smp", config=model_config, name=args.name)
    wandb.config.update(args)

# Store maximum number of nodes for each pair (k, n)
# Used by provably powerful graph networks
max_num_nodes = {4: {12: 12, 20: 20, 28: 28, 36: 36},
                 6: {20: 25, 31: 38, 42: 52, 56: 65},
                 8: {28: 38, 50: 56, 66: 76, 72: 90}}
# Store the maximum degree for the one-hot encoding
max_degree = {4: {12: 4, 20: 6, 28: 7, 36: 7},
              6: {20: 4, 31: 6, 42: 8, 56: 7},
              8: {28: 4, 50: 6, 66: 7, 72: 8}}


def train(epoch):
    """ Train for one epoch. """
    model.train()
    lr_scheduler(args.lr, epoch, optimizer)
    loss_all = 0
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
if args.one_hot:
    max_degree = max_degree[args.k][args.n]
    transform = OneHotDegree(max_degree, cat=False)
    model_config['num_input_features'] = max_degree + 1
    if args.n == 28:
        transform = OneHotDegree(max_degree=7, cat=False)
        model_config['num_input_features'] = 8
elif args.identifiers:
    transform = EyeTransform()
    model_config['num_input_features'] = args.n
elif args.random:
    transform = RandomId()
    model_config['num_input_features'] = 1
else:
    transform = None
    model_config['num_input_features'] = 1


start = time.time()

model_config['num_layers'] = args.k

if model_name == 'SMP':
    model_config['use_batch_norm'] = args.k > 6 or args.n > 30
    model = SMP(**model_config).to(device)
if model_name == 'PPGN':
    transform = DenseAdjMatrix(max_num_nodes[args.k][args.n])
    model_config.pop('num_input_features', None)
    model_config.pop('use_x', None)
    model_config.pop('use_u', None)
    model = ppgn.Powerful(**model_config).to(device)
if model_name == 'GIN':
    model_config['use_batch_norm'] = args.k > 6 or args.n > 50
    model_config.pop('use_x', None)
    model_config.pop('hidden_u', None)
    model = GIN(**model_config).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay=args.weight_decay)
# Load the data
print("Transform used:", transform)
batch_size = args.batch_size
train_data = FourCyclesDataset(args.k, args.n, rootdir, proportion=args.proportion, train=True, transform=transform)
test_data = FourCyclesDataset(args.k, args.n, rootdir, proportion=args.proportion, train=False, transform=transform)
train_loader = DataLoader(train_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, shuffle=True)

print("Starting to train")
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
