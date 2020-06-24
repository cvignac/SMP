from __future__ import division
from __future__ import print_function
import yaml
from multi_task_utils.train import execute_train, build_arg_parser

# Training settings
parser = build_arg_parser()
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--clip', type=float, default=5)
parser.add_argument('--name', type=str, help="name for weights and biases")
parser.add_argument('--debug', action='store_true')
parser.add_argument('--load-from-epoch', type=int, default=-1)
args = parser.parse_args()

yaml_file = 'config_multi_task.yaml'
with open(yaml_file) as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)
    print(model_config)

model_name = model_config['model_name']
model_config.pop('model_name')
print("Model name:", model_name)

if args.wandb or args.name:
    import wandb
    args.wandb = True
    if args.name is None:
        args.name = model_name + f'_{args.k}_{args.n}'
    wandb.init(project="pna_v2", config=model_config, name=args.name)
    wandb.config.update(args)

execute_train(gnn_args=model_config, args=args)
