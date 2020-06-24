# SMP

This folder contains the source code used for Structural Message passing for two tasks:
  - Cycle detection
  - The multi-task regression of graph properties presented in [https://arxiv.org/abs/2004.05718](https://arxiv.org/abs/2004.05718)

Source code for the second task is adapted from [https://github.com/lukecavabarrett/pna](https://github.com/lukecavabarrett/pna).



Pretrained models and more comments will be added in the coming weeks.



## Dependencies
[https://pytorch-geometric.readthedocs.io/en/latest/](Pytorch geometric) v1.5.0 was used. Please follow the instructions on the
website, as simple installations via pip do not work. In particular, the version of pytorch used must match the one of torch-geometric.

Then install the other dependencies:
```
pip install -r requirements.txt
```

## Dataset generation

### Cycle detection
First, download the data from https://1drv.ms/u/s!AolX0ZTx8B-GaoNMtZVmRvsUASs?e=cmtwIm
and unzip it in data/datasets_kcycle_nsamples=10000. Then, run

```
python3 datasets_generation/build_cycles.py
```

### Multi-task regression
Simply run
```
python -m datasets_generation.multitask_dataset
```

## Train

### Cycle detection

In order to train SMP, specify the cycle length, the size of the graphs that is used, and potentially the proportion of the training data
that is kept. For example,
```
python3 cycle_main.py --k 4 --n 12 --proportion 1.0
```
will train the 4-cycle on graph with on average 12 nodes on 1.0 * 100 = 100% of the training data.

In order to run another model, modify models.config.yaml. To run a MPNN that has the
same architecture as SMP, set use_x=True in this file. 

For MPNN and GIN, transforms can be specified in order to add a one-hot encoding of the node degrees,
 or one-hot identifiers. The available options can be seen by using
```
python3 cycles_main.py --help
```

### Multi-task regression

Specify the configuration in the file `config_multi_task.yaml`, and the the available options by using
``` 
python3 multi_task_main.py --help
```
To use default parameters, simply run:
```
python3 multi_task_main.py
```


## License
MIT
