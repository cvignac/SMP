#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ZINC
import argparse
import numpy as np
import time
import yaml
from models.model_zinc import SMPZinc
from models.utils.transforms import OneHotNodeEdgeFeatures

# Change the following to point to the the folder where the datasets are stored
if os.path.isdir('/datasets2/'):
    rootdir = '/datasets2/ZINC/'
else:
    rootdir = './data/ZINC/'
yaml_file = './config_zinc.yaml'

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--wandb', action='store_true',
                    help="Use weights and biases library")
parser.add_argument('--gpu', type=int, help='Id of gpu device. By default use cpu')
parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate")
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--weight-decay', type=float, default=1e-6)
parser.add_argument('--clip', type=float, default=10, help="Gradient clipping")
parser.add_argument('--name', type=str, help="Name for weights and biases")
parser.add_argument('--full', action='store_true')
parser.add_argument('--lr-reduce-factor', type=float, default=0.5)
parser.add_argument('--lr_schedule_patience', type=int, default=100)
parser.add_argument('--save-model', action='store_true', help='Save the model after training')
parser.add_argument('--load-model', action='store_true', help='Evaluate a pretrained model')
parser.add_argument('--lr-limit', type=float, default=5e-6, help='Stop training once it is reached')
args = parser.parse_args()

args.subset = not args.full     # Train either on the full dataset or the subset of 10k samples

# Handle the device
use_cuda = args.gpu is not None and torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
else:
    device = "cpu"
args.device = device
args.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print('Device used:', device)

# Load the config file of the model
with open(yaml_file) as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)
    print(model_config)
    model_config['num_input_features'] = 28 if model_config['use_x'] else 29
    model_config['num_edge_features'] = 3
    model_config['num_classes'] = 1


# Create a name for weights and biases
model_name = 'Zinc_SMP'
if args.name:
    args.wandb = True
if args.wandb:
    import wandb
    if args.name is None:
        args.name = model_name + \
                    f"_{model_config['num_layers']}_{model_config['hidden']}_{model_config['hidden_final']}"
    wandb.init(project="smp-zinc-subset" if args.subset else "smp-zinc", config=model_config, name=args.name)
    wandb.config.update(args)


# The paths can be changed here
if args.save_model or args.load_model:
    if os.path.isdir('/SCRATCH2/'):
        savedir = '/SCRATCH2/vignac/SMP/saved_models/ZINC/'
    else:
        savedir = './saved_models/ZINC/'
    if not os.path.isdir(savedir):
        os.makedirs(savedir)


def train():
    """ Train for one epoch. """
    model.train()
    loss_all = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fct(output, data.y)
        loss.backward()
        loss_all += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    total_mae = 0.0
    for data in loader:
        data = data.to(device)
        output = model(data)
        total_mae += loss_fct(output, data.y).item()
    average_mae = total_mae / len(loader.dataset)
    return average_mae


start = time.time()

model = SMPZinc(**model_config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=args.lr_reduce_factor,
                                                       patience=args.lr_schedule_patience,
                                                       verbose=True)
lr_limit = args.lr_limit

if args.load_model:
    model = torch.load(savedir + model_name + '.pkl')

pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters", pytorch_total_params)

loss_fct = nn.L1Loss(reduction='sum')

# Load the data
batch_size = args.batch_size
transform = OneHotNodeEdgeFeatures(model_config['num_input_features'] - 1, model_config['num_edge_features'])

train_data = ZINC(rootdir, subset=args.subset, split='train', pre_transform=transform)
val_data = ZINC(rootdir, subset=args.subset, split='val', pre_transform=transform)
test_data = ZINC(rootdir, subset=args.subset, split='test', pre_transform=transform)

train_loader = DataLoader(train_data, batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size, shuffle=False)

print("Starting to train")
for epoch in range(args.epochs):
    if args.load_model:
        break
    epoch_start = time.time()
    tr_loss = train()
    current_lr = optimizer.param_groups[0]["lr"]
    if current_lr < lr_limit:
        break
    duration = time.time() - epoch_start
    print(f'Time:{duration:2.2f} | {epoch:5d} | Train MAE: {tr_loss:2.5f} | LR: {current_lr:.6f}')
    mae_val = test(val_loader)
    scheduler.step(mae_val)
    print(f'MAE on the validation set: {mae_val:2.5f}')
    if args.wandb:
        wandb.log({"Epoch": epoch, "Duration": duration, "Train MAE": tr_loss,
                   "Val MAE": mae_val})

if not args.load_model:
    cur_lr = optimizer.param_groups[0]["lr"]
    print(f'{epoch:2.5f} | Loss: {tr_loss:2.5f} | LR: {cur_lr:.6f} | Val MAE: {mae_val:2.5f}')
    print(f'Elapsed time: {(time.time() - start) / 60:.1f} minutes')
    print('done!')

test_mae = test(test_loader)
print(f"Final MAE on the test set: {test_mae}")
print("Done.")

if args.wandb:
    wandb.run.summary['Final test MAE'] = test_mae

if args.save_model:
    torch.save(model, savedir + model_name + '.pkl')