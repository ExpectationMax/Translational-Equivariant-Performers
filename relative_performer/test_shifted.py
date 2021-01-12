#!/usr/bin/env python3
"""Load a model an evaluate its performance on the test set."""
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Subset, Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
from relative_performer.train import DATA_PATH, GPU_AVAILABLE
from relative_performer.test import (
    get_model_class_and_hparams, get_checkpoint_path)


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.dataset.__getitem__(index))

    def __len__(self):
        return self.dataset.__len__()


def get_suitable_digits(dataset, class_ids, min_shift):
    def is_suitable(instance):
        img, y = instance
        if y not in class_ids:
            return False

        col_with_nonzero_values = torch.any(img[0] != 0., dim=0).int()
        # Get first non zero element from both sides
        n_shifts_left = torch.argmax(col_with_nonzero_values) - 1
        n_shifts_right = torch.argmax(
            col_with_nonzero_values.flip(dims=[0])) - 1

        if n_shifts_left >= min_shift and n_shifts_right >= min_shift:
            return True
        else:
            return False

    valid_instances = [is_suitable(dataset[i]) for i in range(len(dataset))]
    valid_indices = np.where(valid_instances)[0]
    return valid_indices


def get_shifting_dataset(dataset_name, class_id, min_shift):
    """Get the dataloader for the test split from the hyperparameters."""
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        dataset = MNIST(
            DATA_PATH.joinpath(dataset_name),
            train=False,
            transform=transform
        )
    elif dataset_name == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        dataset = FashionMNIST(
            DATA_PATH.joinpath(dataset_name),
            train=False,
            transform=transform
        )
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not implemented')
    suitable_digits = get_suitable_digits(dataset, class_id, min_shift)
    print(
        f'{len(suitable_digits)} of {len(dataset)} images are in line with the'
        f' criteria class_id={class_id}, min_shift={min_shift}'
    )
    return Subset(dataset, suitable_digits)


class ShiftingTransform:
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, instance):
        x, y = instance
        if self.shift > 0:
            # Shift image to right
            leftover_image_size = x.shape[2] - self.shift
            # Split along columns
            rest_of_image, cut_off_pixels = torch.split(
                x, [leftover_image_size, self.shift], dim=-1)
            assert torch.all(cut_off_pixels == 0)
            # Glue pixels back to other end of image
            return torch.cat([cut_off_pixels, rest_of_image], dim=-1), y
        else:
            # Shift image to left
            leftover_image_size = x.shape[2] + self.shift  # shift is negative
            # Split along columns (torchvision order is C, H, W)
            cut_off_pixels, rest_of_image = torch.split(
                x, [-self.shift, leftover_image_size], dim=-1)
            assert torch.all(cut_off_pixels == 0)
            # Glue pixels back to other end of image
            return torch.cat([rest_of_image, cut_off_pixels], dim=-1), y


def test_run(directory: Path, class_ids, min_shift):
    model_cls, hparams = get_model_class_and_hparams(directory)
    dataset = get_shifting_dataset(hparams['dataset'], class_ids, min_shift)
    checkpoint = get_checkpoint_path(directory)

    model = model_cls.load_from_checkpoint(checkpoint, **hparams)
    model = model.eval()
    if GPU_AVAILABLE:
        model = model.to('cuda')
    loss_obj = nn.CrossEntropyLoss(reduction='none')

    def accuracy(logits, y):
        y_hat = torch.max(logits, dim=-1).indices
        return (y_hat == y).detach().cpu().float()

    def loss(logits, y):
        return loss_obj(logits, y).detach().cpu()

    metrics = {
        'accuracy': accuracy,
        'loss': loss
    }
    summary_statistics = {
        'mean': torch.mean,
        'std': torch.std
    }

    results = []
    with torch.no_grad():
        # tqdm.trange(-min_shift, min_shift+1, position=0, desc='shifts'):
        for shift in [3, 4, 5]:
            cur_res = {'shift': shift}
            shifted_data = TransformedDataset(
                dataset, ShiftingTransform(shift))
            data_loader = DataLoader(
                shifted_data,
                hparams['batch_size'],
                num_workers=0,
                pin_memory=True
            )

            metrics_out = defaultdict(lambda: [])
            for batch in tqdm.tqdm(
                    data_loader, total=len(data_loader), position=1,
                    desc='batches', leave=False):
                x, y = batch
                if GPU_AVAILABLE:
                    x = x.to('cuda')
                    y = y.to('cuda')
                x = x.permute(0, 2, 3, 1)

                logits = model(x)
                for metric_name, metric_fn in metrics.items():
                    metrics_out[metric_name].append(metric_fn(logits, y))

            # Combine arrays and compute summary statistics
            metrics_out = {
                key: torch.cat(values, dim=-1)
                for key, values in metrics_out.items()
            }
            for metric_name, values in metrics_out.items():
                for stat_name, fn in summary_statistics.items():
                    cur_res[f'{metric_name}_{stat_name}'] = float(fn(values))
            results.append(cur_res)
    return pd.DataFrame(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dirs', type=Path, nargs='+')
    parser.add_argument('--labels', type=int, default=[1], nargs='+')
    parser.add_argument('--min_shift', type=int, default=10)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    results = []
    for run_dir in args.run_dirs:
        print(run_dir)
        result = test_run(run_dir, args.labels, args.min_shift)
        result['run_dir'] = run_dir
        results.append(result)
    results = pd.concat(results, ignore_index=True)
    results = results[
        ['run_dir'] + [col for col in results.columns if col != 'run_dir']]
    results.to_csv(args.output)
