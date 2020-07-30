#!/usr/bin/env python
# coding: utf-8
import os
import torch
from itertools import product
import argparse
import numpy as np
import yaml
from models.smp_cycles import SMP
from train_eval_tu import cross_validation_with_val_set
from utils.tu_datasets import get_dataset

# Change the following to point to the the folder where the datasets are stored
if os.path.isdir('/datasets2/'):
    rootdir = '/datasets2/TU_datasets/'
else:
    rootdir = './data/TU_datasets/'
yaml_file = './config_tu.yaml'
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--save-model', action='store_true',
                    help='Save the model once training is done')
parser.add_argument('--wandb', action='store_true',
                    help="Use weights and biases library")
parser.add_argument('--gpu', type=int, help='Id of gpu device. By default use cpu')
parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate")
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--weight-decay', type=float, default=0.000001)
parser.add_argument('--clip', type=float, default=10, help="Gradient clipping")
parser.add_argument('--name', type=str, help="Name for weights and biases")
parser.add_argument('--lr_decay_factor', type=float, default=0.9)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
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


# Create a name for weights and biases
if args.wandb:
    import wandb
    if args.name is None:
        args.name = f'SMP_{model_config["num_layers"]}_{model_config["hidden"]}_{model_config["hidden_final"]}'
    wandb.init(project="smp-classif", config=model_config, name=args.name)
    wandb.config.update(args)

# Load the data
dataset_name = model_config.pop('dataset_name', None)
dataset = get_dataset(dataset_name, model_config['num_layers'], sparse=False)


def run_config(model_config):
    print(model_config)
    model = SMP(**model_config, num_input_features= 1 + dataset.num_node_features,
                num_classes=dataset.num_classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay=args.weight_decay)

    def logger(info):
        fold, epoch = info['fold'] + 1, info['epoch']
        val_loss, test_acc = info['val_loss'], info['test_acc']
        print('{:02d}/{:03d}: Train Loss: {:.4f}, Val Loss: {:.4f}, Test Accuracy: {:.3f}, Learning Rate: {:.5f}'.format(
            fold, epoch, info['train_loss'], val_loss, test_acc, info['lr']))

    best_epoch, mean_acc_best_epoch, final_std = cross_validation_with_val_set(
                dataset, model, folds=10,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_decay_factor=args.lr_decay_factor,
                lr_decay_step_size=args.lr_decay_step_size,
                weight_decay=0,
                use_wandb=args.wandb,
                device=device,
                logger=logger)
    print("Done.")
    return best_epoch, mean_acc_best_epoch, final_std


n_layers = [1, 3]
hidden_f = [64, 128]

results = []
for layers, hidden in product(n_layers, hidden_f):
    model_config['num_layers'] = layers
    model_config['hidden_final'] = hidden
    epoch, acc, std = run_config(model_config)
    results.append([layers, hidden, acc, std, epoch])

results = np.array(results)
if args.wandb:
    wandb.run.summary["Final results"] = results

print("Dataset:", dataset_name)
for res in results:
    print("{} layers - {} hidden final: Average acc = {} +- {}, best epoch at {}".format(*res))





