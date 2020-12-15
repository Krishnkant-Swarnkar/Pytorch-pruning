# Pytorch-Pruning

This repository contains the implementations of neural network weight pruning methods.

## Available Pruning Methods

Pruning has three stages:

1. Training
2. Pruning:
   - OneShotPruning
3. Retraining
   - No Retraining (none)
   - FineTuning (fine-tuning)
   - Learning Rate Rewinding (lr-rewinding)
   - Weight Rewinding (weight-rewinding)

## Usage

Install the following python libraries:
``torch>=1.6.0
torchvision>=0.7.0
numpy
tqdm
livelossplot``
Run :
`$ mkdir saved_models`

To get the pruned models you need:
1. A Model (torch.nn.module)
   - for which forward() takes input as a single variable "batch"
   - which returns loss function (by default) when the forward() is called, and returns the prediction when get_prediction argument is set to true in the forward.
2. Torch Data Loaders (train_dataloader, val_dataloader)

See the Jupyter Notebooks (.ipynb files) for a better idea about how to do the pruning. This repository contains implementations of the following Models:
`lenet, resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202` and functions to load data loaders for `MNIST, CIFAR-10`




## Directory Structure

	.
	|--- models
	|    |--- __init__.py
	|    |--- lenet.py
	|    |--- resnet.py
	|--- pruning
	|    |--- unstructured
	|    |    |--- __init__.py
	|    |    |--- one_shot_pruning.py
	|    |    |--- iterative_pruning.py
	|    |--- structured
	|    |    |--- __init__.py
	|--- utils
	|    |--- __init__.py
	|    |--- data_utils.py
	|    |--- train_utils.py
	|--- Lisence
	|--- README.md
	|--- LISENCE


## Todo

1. Write code for weight distribution visualizations for each layer.
2. Write code for iterative pruning.
3. Write code for ADMM sparsity regularization.







