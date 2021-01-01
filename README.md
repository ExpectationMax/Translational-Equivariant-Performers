# RelativePerformer
Relative Encoding of `Performer` architecture.

## Installation
Install [poetry](https://python-poetry.org/docs/#installation) in a python
environment with python version larger than $3.7$.
Alternatively use [pyenv](https://github.com/pyenv/pyenv) to install an
appropriate python version.  Pyenv and poetry are compatible, for further
details check out the section on pyenv in the [poetry
documentation](https://python-poetry.org/docs/managing-environments/).

Then install the package in a new virtual environment using
```bash
poetry install --dev
```

You can then enter the virtual environment using `poetry shell` or run commands
inside the virtual environment using `poetry run <command>`.

## Usage
The main script for training models is `relative_performer/train.py`, it allows
to define training and model parameters and the dataset that should be used to
train the model.

```bash
$ poetry run relative_performer/train.py --help

usage: train.py [-h] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
                [--dim DIM] [--depth DEPTH] [--heads HEADS]
                [--pos_scales POS_SCALES]
                {FashionMNIST,MNIST,CIFAR10,TinyCIFAR10}

positional arguments:
  {FashionMNIST,MNIST,CIFAR10,TinyCIFAR10}

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
  --learning_rate LEARNING_RATE
  --dim DIM
  --depth DEPTH
  --heads HEADS
  --pos_scales POS_SCALES
```

This script should be extended to also support other types of models.

The logs of successful runs will be stored in the path
`lighting_logs/version_X` where `X` is automatically incremented for each run
and missing datasets will automatically be downloaded to the path
`<project_root>/data`.

### Example - Training on MNIST
Training the relative performer model on MNIST with default parameters can be
achieved using the command below:

```bash
$ poetry run relative_performer/train.py MNIST
GPU available: False, used: False
TPU available: None, using: 0 TPU cores

  | Name                 | Type                      | Params
-------------------------------------------------------------------
0 | positional_embedding | LearnableSinusoidEncoding | 4
1 | content_embedding    | Linear                    | 256
2 | performer            | RelativePerformer         | 793 K
3 | output_layer         | Linear                    | 1.3 K
4 | loss                 | CrossEntropyLoss          | 0
5 | train_acc            | Accuracy                  | 0
6 | val_acc              | Accuracy                  | 0
-------------------------------------------------------------------
795 K     Trainable params
0         Non-trainable params
795 K     Total params
Epoch 0:   1%|▊    | 13/1874 [01:07<2:40:57,  5.19s/it, loss=2.72, v_num=11, train_acc_step=0.125]

```


