# Building powerful and equivariant graph neural networks with structural message-passing

This paper contains code for the paper *Building powerful and equivariant graph neural networks with structural message-passing* (Neurips 2020) by
[Clément Vignac](https://cvignac.github.io/), [Andreas Loukas](https://andreasloukas.blog/) and [Pascal Frossard](https://www.epfl.ch/labs/lts4/people/people-current/frossard/).
[Link to the paper](https://papers.nips.cc/paper/2020/file/a32d7eeaae19821fd9ce317f3ce952a7-Paper.pdf)

Abstract:

Message-passing has proved to be an effective way to design graph neural networks,
as it is able to leverage both permutation equivariance and an inductive bias towards
learning local structures in order to achieve good generalization. However, current
message-passing architectures have a limited representation power and fail to learn
basic topological properties of graphs. We address this problem and propose a
powerful and equivariant message-passing framework based on two ideas: first,
we propagate a one-hot encoding of the nodes, in addition to the features, in order
to learn a local context matrix around each node. This matrix contains rich local
information about both features and topology and can eventually be pooled to build
node representations. Second, we propose methods for the parametrization of the
message and update functions that ensure permutation equivariance. Having a
representation that is independent of the specific choice of the one-hot encoding
permits inductive reasoning and leads to better generalization properties. Experi-
mentally, our model can predict various graph topological properties on synthetic
data more accurately than previous methods and achieves state-of-the-art results on
molecular graph regression on the ZINC dataset.

## Code overview


This folder contains the source code used for Structural Message passing for three tasks:
  - Cycle detection
  - The multi-task regression of graph properties presented in [https://arxiv.org/abs/2004.05718](https://arxiv.org/abs/2004.05718)
  - Constrained solubility regression on ZINC

Source code for the second task is adapted from [https://github.com/lukecavabarrett/pna](https://github.com/lukecavabarrett/pna).


## Dependencies
[https://pytorch-geometric.readthedocs.io/en/latest/](Pytorch geometric) v1.6.1 was used. Please follow the instructions on the
website, as simple installations via pip do not work. In particular, the version of pytorch used must match the one of torch-geometric.

Then install the other dependencies:
```
pip install -r requirements.txt
```

## Dataset generation

### Cycle detection
First, download the data from https://drive.switch.ch/index.php/s/hv65hmY48GrRAoN
and unzip it in data/datasets_kcycle_nsamples=10000. Then, run

```
python3 datasets_generation/build_cycles.py
```

### Multi-task regression
Simply run
```
python -m datasets_generation.multitask_dataset
```

### ZINC
We use the pytorch-geometric downloader, there should be nothing to to by hand.
## Folder structure

  - Each task is launched by running the corresponding *main* file (cycles_main, zinc_main, multi_task_main).
  - The model parameters can be changed in the associated config.yaml file, while training parameters are modified
with command line arguments. 
  - The model used for each task is located in the model folder (model_cycles,
model_multi_task, model_zinc). 
  - They all use some of the SMP layers parametrized in the smp_layers file. 
  - All SMP layers use the same set of base functions in models/utils/layers.py. These functions map tensors of one order
to tensors of another order using a predefined set of equivariant transformations.

## Train

### Cycle detection

In order to train SMP, specify the cycle length, the size of the graphs that is used, and potentially the proportion of the training data
that is kept. For example,
```
python3 cycle_main.py --k 4 --n 12 --proportion 1.0 --gpu 0
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
python3 multi_task_main.py --gpu 0
```

### ZINC

The ZINC dataset is downloaded through pytorch geometric, but the destination folder should be specified at 
the beginning of `zinc_main.py`. Model parameters can be changed in `config_zinc.yaml`. 

To use default parameters, simply run:
```
python3 zinc_main.py --gpu 0
```

## Use SMP on new data

This code is currently not available as a library, so you will need to copy-paste files to adapt it to your 
own data. 
While most of the code can be reused, you may need to adapt the model to your own problem. We advise you to look at the
different model files (model_cycles, model_multi_task, model_zinc) to see how they are built. They all follow the same
design:
  - A local context is first created using the functions in models.utils.misc. If you have node features that
  you wish to use in SMP, use `map_x_to_u` to include them in the local contexts.
  - One of the three SMP layers (SMP, FastSMP, SimplifiedFastSMP) is used at each layer to update the local context.
  Then either some node-level features or some graph-level features are extracted. For this purpose, you can use
  the `NodeExtractor` and `GraphExtractor` classes in `models.utils.layers.py`.
  - The extracted features are processed by a standard neural network. You can use a multi-layer perceptron here, or
  a more complex structure such as a Gated Recurrent Network that will take as input the features extracted at
  each layer.
  
To sum up, you need to copy the following files to your own folder:
  - models.smp_layers.py
  - models.utils.layers.py and models.utils.misc.py

and to adapt the following files to your own problem:
  - the main file (e.g. zinc_main.py)
  - the config file (e.g. config_zinc.yaml)
  - the model file (e.g. models/model_zinc.py)
  
We advise you to use the "weights and biases" library as well, as we found it very convenient to store results.

## License
MIT

## Cite this paper

@inproceedings{NEURIPS2020_a32d7eea,
 author = {Vignac, Cl\'{e}ment and Loukas, Andreas and Frossard, Pascal},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {14143--14155},
 publisher = {Curran Associates, Inc.},
 title = {Building powerful and equivariant graph neural networks with structural message-passing},
 url = {https://proceedings.neurips.cc/paper/2020/file/a32d7eeaae19821fd9ce317f3ce952a7-Paper.pdf},
 volume = {33},
 year = {2020}
}


