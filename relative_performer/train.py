#!/usr/bin/env python3
"""Train a model."""
import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pl_bolts.datamodules as datasets
from einops import rearrange

from relative_performer.performer_pytorch import Performer
from relative_performer.constrained_relative_encoding import (
    RelativePerformer, LearnableSinusoidEncoding)

GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0
DATA_PATH = Path(__file__).parent.parent.joinpath('data')



class TbWithMetricsLogger(pl.loggers.TensorBoardLogger):
    def __init__(self, save_dir, initial_values, **kwargs):
        super().__init__(save_dir, default_hp_metric=False, **kwargs)
        self.hparams_saved = False
        self.initial_values = initial_values

    @pl.utilities.rank_zero_only
    def log_hyperparams(self, params):
        print('Called log hparams')
        # Somehow hyperparameters are saved when a model is simply restored,
        # catch that here so we don't add an incorrect value when restoring.
        if self.hparams_saved:
            return
        super().log_hyperparams(
            params,
            self.initial_values
        )
        self.hparams_saved = True


class PerfomerBase(pl.LightningModule):
    """Base class of Performer models for image classification."""

    positional_embedding: nn.Module

    def __init__(self, in_features, dim, num_classes, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.dim = dim
        self.num_classes = num_classes

        self.content_embedding = nn.Linear(in_features, dim)
        self.class_query = nn.Parameter(torch.Tensor(dim))
        self.output_layer = nn.Linear(dim, num_classes)
        self.loss = nn.CrossEntropyLoss()

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.class_query)

    def _flatten_to_sequence(self, input: torch.Tensor):
        """Flatten the 2D input into a 1D sequence.

        Preserve positional information in separate tensor.

        Args:
            input (torch.Tensor): Embeddings [bs, nx, ny, d]

        Returns:
            embeddings [bs, nx*ny, d], positions [bs, nx*ny, 2]
        """
        device, dtype = input.device, input.dtype
        nx, ny = input.shape[1:3]
        x_pos = torch.arange(0, nx, device=device, dtype=dtype)
        y_pos = torch.arange(0, ny, device=device, dtype=dtype)
        positions = torch.stack(torch.meshgrid(x_pos, y_pos), axis=-1)
        del x_pos, y_pos
        return (
            rearrange(input, 'b x y d -> b (x y) d'),
            rearrange(positions, 'x y d -> 1 (x y) d')
        )

    def _compute_positional_embeddings(self, positions):
        """Compute positional embeddings."""
        return rearrange(
            self.positional_embedding(positions), 'b n p d -> b n (p d)')

    def _add_class_query(self, embedding, pos_embedding):
        """Add class query element to beginning of sequences.

        Args:
            embedding: The element embeddings
            pos_embedding: The positional embedding

        Returns:
            embeddings, pos_embeddings both with additional class query element
            at the beginning of the sequence.
        """
        device = embedding.device
        bs_embedding = embedding.shape[0]
        bs_pos, *_, pos_embedding_dim = pos_embedding.shape
        # Add learnt class query to input, with zero positional encoding
        embedding = torch.cat(
            [
                self.class_query[None, None, :].expand(bs_embedding, -1, -1),
                embedding
            ],
            axis=1
        )
        pos_embedding = torch.cat(
            [
                torch.zeros(1, 1, pos_embedding_dim, device=device),
                pos_embedding
            ],
            axis=1
        )
        return embedding, pos_embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        # The datasets always input in the format (C, W, H) instead of (W, H,
        # C).
        x = x.permute(0, 2, 3, 1)
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        self.train_acc(logits, y)
        self.log('train/acc', self.train_acc, on_step=True, on_epoch=True,
                 prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # The datasets always input in the format (C, W, H) instead of (W, H,
        # C).
        x = x.permute(0, 2, 3, 1)
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('val/loss', loss, on_epoch=True)
        self.val_acc(logits, y)
        self.log('val/acc', self.val_acc, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        # The datasets always input in the format (C, W, H) instead of (W, H,
        # C).
        x = x.permute(0, 2, 3, 1)
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('test/loss', loss, on_epoch=True)
        self.test_acc(logits, y)
        self.log('test/acc', self.test_acc, on_epoch=True)
        return loss


class PerformerModel(PerfomerBase):
    def __init__(self, dim, depth, heads, max_pos=32, **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.save_hyperparameters()
        self.positional_embedding = LearnableSinusoidEncoding(
            dim // 2, max_timescale_init=max_pos*1000)
        self.performer = Performer(
            dim,
            depth,
            heads
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.hparams.learning_rate)

    def forward(self, x):
        embedding = self.content_embedding(x)
        embedding, positions = self._flatten_to_sequence(embedding)
        positions = self._compute_positional_embeddings(positions)
        embedding, positions = self._add_class_query(embedding, positions)
        embedding += positions
        del positions
        # First element contains class prediction
        out = self.performer(embedding)[:, 0]
        return self.output_layer(out)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--dim', type=int, default=128)
        parser.add_argument('--depth', type=int, default=4)
        parser.add_argument('--heads', type=int, default=4)
        return parser


class NoposPerformerModel(PerfomerBase):
    def __init__(self, dim, depth, heads, max_pos=32, **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.save_hyperparameters()
        self.performer = Performer(
            dim,
            depth,
            heads
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.hparams.learning_rate)

    def forward(self, x):
        embedding = self.content_embedding(x)
        embedding = rearrange(embedding, 'b x y d -> b (x y) d')
        bs_embedding = embedding.shape[0]
        # Add learnt class query to input, with zero positional encoding
        embedding = torch.cat(
            [
                self.class_query[None, None, :].expand(bs_embedding, -1, -1),
                embedding
            ],
            axis=1
        )
        # First element contains class prediction
        out = self.performer(embedding)[:, 0]
        return self.output_layer(out)

    @ staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--dim', type=int, default=128)
        parser.add_argument('--depth', type=int, default=4)
        parser.add_argument('--heads', type=int, default=4)
        return parser


class RelativePerformerModel(PerfomerBase):
    def __init__(self, dim, depth, heads, pos_scales, pos_dims=1, max_pos=32,
                 **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.save_hyperparameters()
        self.positional_embedding = LearnableSinusoidEncoding(
            pos_scales*2, max_timescale_init=max_pos*2)
        self.performer = RelativePerformer(
            dim,
            depth,
            heads,
            pos_dims=pos_dims,
            pos_scales=pos_scales
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.hparams.learning_rate)

    def forward(self, x):
        embedding = self.content_embedding(x)
        embedding, positions = self._flatten_to_sequence(embedding)
        positions = self._compute_positional_embeddings(positions)
        # First element contains class prediction
        embedding, positions = self._add_class_query(embedding, positions)
        out = self.performer(embedding, positions)[:, 0]
        return self.output_layer(out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # The datasets always input in the format (C, W, H) instead of (W, H,
        # C).
        x = x.permute(0, 2, 3, 1)
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True,
                 prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # The datasets always input in the format (C, W, H) instead of (W, H,
        # C).
        x = x.permute(0, 2, 3, 1)
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.val_acc(logits, y)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        return loss

    @ staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--dim', type=int, default=128)
        parser.add_argument('--depth', type=int, default=4)
        parser.add_argument('--heads', type=int, default=4)
        parser.add_argument('--pos_scales', type=int, default=4)
        return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['Performer', 'RelativePerformer',
                                          'NoposPerformer'])
    parser.add_argument('dataset', choices=[
        'FashionMNIST', 'MNIST', 'CIFAR10'])
    parser.add_argument('--log_path', type=str, default='lightning_logs')
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--version', type=str, default=None)

    parser.add_argument('--batch_size', default=16, type=int)

    partial_args, _ = parser.parse_known_args()
    if partial_args.model == 'Performer':
        parser = PerformerModel.add_model_specific_args(parser)
        model_cls = PerformerModel
    elif partial_args.model == 'RelativePerformer':
        parser = RelativePerformerModel.add_model_specific_args(parser)
        model_cls = RelativePerformerModel
    elif partial_args.model == 'NoposPerformer':
        parser = NoposPerformerModel.add_model_specific_args(parser)
        model_cls = NoposPerformerModel
    args = parser.parse_args()

    data_cls = getattr(datasets, args.dataset + 'DataModule')
    num_workers = 0
    # Handle incosistencies in DataModules: Some datasets accept batch_size,
    # some don't, some simply ignore it.
    if args.dataset == 'MNIST':
        dataset = datasets.MNISTDataModule(
            DATA_PATH.joinpath(args.dataset),
            num_workers=num_workers
        )
    elif args.dataset == 'FashionMNIST':
        dataset = datasets.FashionMNISTDataModule(
            DATA_PATH.joinpath(args.dataset),
            num_workers=num_workers
        )
    elif args.dataset == 'CIFAR10':
        dataset = datasets.CIFAR10DataModule(
            DATA_PATH.joinpath(args.dataset),
            num_workers=num_workers,
            batch_size=args.batch_size
        )

    dataset.prepare_data()

    in_features, nx, ny = dataset.dims
    max_pos = max(nx, ny)
    model = model_cls(
        **vars(args),
        in_features=in_features,
        pos_dims=2,
        max_pos=max_pos,
        num_classes=dataset.num_classes
    )

    # Setup logging, checkpointing and early stopping
    logger = TbWithMetricsLogger(
        args.log_path,
        {
            'train/loss_epoch': float('inf'),
            'train/acc_epoch': float('-inf'),
            'val/loss': float('inf'),
            'val/acc': float('-inf'),
            'test/loss': float('inf'),
            'test/acc': float('-inf')
        },
        name=args.exp_name,
        version=args.version
    )
    model_checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='val/acc',
        mode='max',
        save_top_k=1,
        dirpath=os.path.join(logger.log_dir, 'checkpoints')
    )
    early_stopping_cb = pl.callbacks.EarlyStopping(
        monitor='val/acc', patience=10, mode='max', strict=True,
        verbose=1)

    trainer = pl.Trainer(
        gpus=-1 if GPU_AVAILABLE else None,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[model_checkpoint_cb, early_stopping_cb],
        limit_train_batches=2, limit_val_batches=1, limit_test_batches=1, max_epochs=2)
    # Handle incosistencies in DataModules: Some datasets only listen to the
    # batch_size argumen if it is passed here, others don't have to argument.
    # MNIST and FashionMNIST take batch_size as argument here, while CIFAR10
    # requires it as a constructor argument.
    try:
        train_loader = dataset.train_dataloader(batch_size=args.batch_size)
        val_loader = dataset.val_dataloader(batch_size=args.batch_size)
        test_loader = dataset.test_dataloader(batch_size=args.batch_size)
    except TypeError:
        train_loader = dataset.train_dataloader()
        val_loader = dataset.val_dataloader()
        test_loader = dataset.test_dataloader()

    trainer.fit(
        model,
        train_dataloader=train_loader,
        val_dataloaders=val_loader,
    )
    print('Loading model form', model_checkpoint_cb.best_model_path)
    trainer.test(
        ckpt_path=model_checkpoint_cb.best_model_path,
        test_dataloaders=test_loader
    )
    logger.save()
