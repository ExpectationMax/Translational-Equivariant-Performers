#!/usr/bin/env python3
"""Load a model an evaluate its performance on the test set."""
import argparse
from pathlib import Path

from yaml import load, Loader
import pandas as pd

import torchvision.transforms as transforms
import pytorch_lightning as pl
import pl_bolts.datamodules as datasets
import relative_performer.train as train_module
from relative_performer.train import DATA_PATH, ToIntTensor


class NoCheckpointFoundException(Exception):
    pass


def get_test_dataloader(hparams):
    """Get the dataloader for the test split from the hyperparameters."""
    num_workers = 4
    # Handle incosistencies in DataModules: Some datasets accept batch_size,
    # some don't, some simply ignore it.
    if hparams['dataset'] == 'MNIST':
        if hparams['embedding_type'] == 'lookup':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                ToIntTensor
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])

        dataset = datasets.MNISTDataModule(
            DATA_PATH.joinpath(hparams['dataset']),
            num_workers=num_workers,
            train_transforms=transform,
            val_transforms=transform,
            test_transforms=transform
        )
    elif hparams['dataset'] == 'FashionMNIST':
        if hparams['embedding_type'] == 'lookup':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                ToIntTensor
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])
        dataset = datasets.FashionMNISTDataModule(
            DATA_PATH.joinpath(hparams['dataset']),
            num_workers=num_workers,
            train_transforms=transform,
            val_transforms=transform,
            test_transforms=transform
        )
    elif hparams['dataset'] == 'CIFAR10':
        if hparams['embedding_type'] == 'lookup':
            transform = ToIntTensor
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        dataset = datasets.CIFAR10DataModule(
            DATA_PATH.joinpath(hparams['dataset']),
            num_workers=num_workers,
            batch_size=hparams['batch_size'],
            train_transforms=transform,
            val_transforms=transform,
            test_transforms=transform
        )
    try:
        test_loader = dataset.test_dataloader(batch_size=hparams['batch_size'])
    except TypeError:
        test_loader = dataset.test_dataloader()
    return test_loader


def get_model_class_and_hparams(directory: Path):
    """Get the model class used to generate a run."""
    with directory.joinpath('hparams.yaml').open('r') as f:
        hparams = load(f, Loader=Loader)
    model_name = hparams['model']
    model_cls = getattr(train_module, model_name + 'Model')

    # Handle cases when run was created before extensions
    if 'embedding_type' not in hparams.keys():
        hparams['embedding_type'] = 'linear'
    if 'schedule' not in hparams.keys():
        hparams['schedule'] = 'constant'
    if 'warmup' not in hparams.keys():
        hparams['warmup'] = 0
    if 'content_rel_attn' not in hparams.keys():
        hparams['content_rel_attn'] = False

    return model_cls, hparams


def get_checkpoint_path(directory: Path):
    checkpoints = list(directory.joinpath('checkpoints').glob('*.ckpt'))
    if len(checkpoints) == 0:
        raise NoCheckpointFoundException()
    return checkpoints[0]


def test_run(directory: Path):
    model_cls, hparams = get_model_class_and_hparams(directory)
    test_loader = get_test_dataloader(hparams)
    checkpoint = get_checkpoint_path(directory)

    model = model_cls.load_from_checkpoint(checkpoint, **hparams)
    trainer = pl.Trainer(logger=False)
    test_results = trainer.test(model, test_dataloaders=test_loader)[0]
    result = {**hparams, **{key: float(value)
                            for key, value in test_results.items()}}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    result = test_run(args.run_dir)
    pd.Series(result).to_csv(args.output)
