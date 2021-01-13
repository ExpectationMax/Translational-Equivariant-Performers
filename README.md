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

usage: train.py [-h] [--log_path LOG_PATH] [--exp_name EXP_NAME]
                [--version VERSION] [--batch_size BATCH_SIZE]
                [--embedding_type {linear,MLP,lookup}]
                {Performer,RelativePerformer,NoposPerformer,ClippedRelativePerformer}
                {FashionMNIST,MNIST,CIFAR10}

positional arguments:
  {Performer,RelativePerformer,NoposPerformer,ClippedRelativePerformer}
                        The model to train
  {FashionMNIST,MNIST,CIFAR10}
                        The dataset to train on

optional arguments:
  -h, --help            show this help message and exit
  --log_path LOG_PATH   Logging path
  --exp_name EXP_NAME   Experiment name
  --version VERSION     Version of experiment
  --batch_size BATCH_SIZE
                        Batch size used for training.
  --embedding_type {linear,MLP,lookup}
                        Embedding type used to embed pixel values (default:
                        linear)
```

The logs of successful runs will be stored in the path
`<log_path>/<exp_name>/<version>`. If `--version` is not explicitly defined it
will automatically be set to the pattern `version_X`  where `X` is
automatically incremented for each run. By default (if no command line
parameters are provided) the path is `lighting_logs/default/version_X`.

Missing datasets will automatically be downloaded to the path
`<project_root>/data`.


### Model specific arguments
Each model can additionally have specific arguments associated which may also
be defined via the command line

```bash
$ poetry run relative_performer/train.py Performer MNIST --help

usage: train.py [-h] [--log_path LOG_PATH] [--exp_name EXP_NAME]
                [--version VERSION] [--batch_size BATCH_SIZE]
                [--embedding_type {linear,MLP,lookup}]
                [--learning_rate LEARNING_RATE] [--warmup WARMUP]
                [--schedule {constant,noam}] [--dim DIM] [--depth DEPTH]
                [--heads HEADS] [--attn_dropout ATTN_DROPOUT]
                [--ff_dropout FF_DROPOUT]
                [--feature_redraw_interval FEATURE_REDRAW_INTERVAL]
                [--no_projection]
                {Performer,RelativePerformer,NoposPerformer,ClippedRelativePerformer}
                {FashionMNIST,MNIST,CIFAR10}

positional arguments:
  {Performer,RelativePerformer,NoposPerformer,ClippedRelativePerformer}
                        The model to train
  {FashionMNIST,MNIST,CIFAR10}
                        The dataset to train on

optional arguments:
  -h, --help            show this help message and exit
  --log_path LOG_PATH   Logging path
  --exp_name EXP_NAME   Experiment name
  --version VERSION     Version of experiment
  --batch_size BATCH_SIZE
                        Batch size used for training.
  --embedding_type {linear,MLP,lookup}
                        Embedding type used to embed pixel values (default:
                        linear)
  --learning_rate LEARNING_RATE
  --warmup WARMUP
  --schedule {constant,noam}
  --dim DIM
  --depth DEPTH
  --heads HEADS
  --attn_dropout ATTN_DROPOUT
  --ff_dropout FF_DROPOUT
  --feature_redraw_interval FEATURE_REDRAW_INTERVAL
  --no_projection
```

Which shows that the `Performer` model additionally supports the arguments
`--learning_rate`, `--dim`, `--depth` and `--heads` etc.

### Example - Training on MNIST
Training the performer model on MNIST with default parameters can be
achieved using the command below:

```bash
$ poetry run relative_performer/train.py MNIST

No correct seed found, seed set to 2111136583
GPU available: False, used: False
TPU available: None, using: 0 TPU cores

  | Name                 | Type                      | Params
-------------------------------------------------------------------
0 | content_embedding    | Linear                    | 256   
1 | output_layer         | Linear                    | 1.3 K 
2 | loss                 | CrossEntropyLoss          | 0     
3 | train_acc            | Accuracy                  | 0     
4 | val_acc              | Accuracy                  | 0     
5 | test_acc             | Accuracy                  | 0     
6 | positional_embedding | LearnableSinusoidEncoding | 32    
7 | performer            | Performer                 | 793 K 
-------------------------------------------------------------------
794 K     Trainable params
0         Non-trainable params
794 K     Total params
Running with random seed: 2111136583
Called log hparams
Validation sanity check: 100%|██████████| 2/2 [00:03<00:00,  1.55s/it]
Training: Epoch 0:   0%|          | 2/3749 [00:10<5:14:18,  5.03s/it, loss=3.22, v_num=38, train/acc_step=0.0625]
```


