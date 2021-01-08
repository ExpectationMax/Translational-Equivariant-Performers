#!/usr/bin/env python3
"""Train a model."""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pytorch_lightning as pl
import pl_bolts.datamodules as datasets
from einops import rearrange

from relative_performer.embedding_utils import (
    LookupEmbedding, MLPEmbedding, ToIntTensor)
from relative_performer.logging_utils import TbWithMetricsLogger
from relative_performer.performer_pytorch import Performer
from relative_performer.clipped_relative_attention import ClippedRelativePerformer
from relative_performer.constrained_relative_encoding import (
    RelativePerformer, LearnableSinusoidEncoding)
from relative_performer.training_utils import (
    get_constant_schedule_with_warmup, get_noam_schedule)

GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0
DATA_PATH = Path(__file__).parent.parent.joinpath('data')


class PerfomerBase(pl.LightningModule):
    """Base class of Performer models for image classification."""

    positional_embedding: nn.Module

    def __init__(self, in_features, dim, num_classes,
                 embedding_type, learning_rate, warmup, schedule, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.dim = dim
        self.num_classes = num_classes

        if embedding_type == 'linear':
            self.content_embedding = nn.Linear(in_features, dim)
        elif embedding_type == 'MLP':
            self.content_embedding = MLPEmbedding(in_features, dim)
        elif embedding_type == 'lookup':
            self.content_embedding = LookupEmbedding(in_features, dim)
        else:
            raise ValueError(
                'Invalid embedding_type: {}'.format(embedding_type))
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

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            self.hparams.learning_rate,
            betas=[0.9, 0.98],
            # weight_decay=0.1
        )
        if self.hparams.warmup != 0:
            if self.hparams.schedule == 'linear':
                scheduler = get_constant_schedule_with_warmup(
                    optim, self.hparams.warmup)
            elif self.hparams.schedule == 'noam':
                scheduler = get_noam_schedule(
                    optim, self.hparams.warmup)
            scheduler = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
            return [optim], [scheduler]
        else:
            return optim

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

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--warmup', default=0, type=int)
        parser.add_argument('--schedule', default='constant',
                            choices=['constant', 'noam'])
        return parser


class PerformerModel(PerfomerBase):
    def __init__(self, dim, depth, heads, attn_dropout=0., ff_dropout=0.,
                 max_pos=32, feature_redraw_interval=100, no_projection=False,
                 **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.save_hyperparameters()
        self.positional_embedding = LearnableSinusoidEncoding(
            dim // 2, max_timescale_init=max_pos*1000)
        self.performer = Performer(
            dim,
            depth,
            heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            feature_redraw_interval=feature_redraw_interval,
            no_projection=no_projection
        )

    def forward(self, x, positions=None):
        embedding = self.content_embedding(x)

        # If positions are provided we assume the input has already been
        # flattened
        if positions is None:
            embedding, positions = self._flatten_to_sequence(embedding)

        positions = self._compute_positional_embeddings(positions)
        embedding, positions = self._add_class_query(embedding, positions)
        embedding += positions
        del positions
        # First element contains class prediction
        out = self.performer(embedding)[:, 0]
        return self.output_layer(out)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument('--dim', type=int, default=128)
        parser.add_argument('--depth', type=int, default=4)
        parser.add_argument('--heads', type=int, default=4)
        parser.add_argument('--attn_dropout', default=0.1, type=float)
        parser.add_argument('--ff_dropout', default=0.1, type=float)
        parser.add_argument('--feature_redraw_interval',
                            type=int, default=1000)
        parser.add_argument(
            '--no_projection', default=False, action='store_true')
        return parser


class NoposPerformerModel(PerfomerBase):
    def __init__(self, dim, depth, heads, attn_dropout=0., ff_dropout=0.,
                 max_pos=32, feature_redraw_interval=100, no_projection=False,
                 **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.save_hyperparameters()
        self.performer = Performer(
            dim,
            depth,
            heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            feature_redraw_interval=feature_redraw_interval,
            no_projection=no_projection
        )

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

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument('--dim', type=int, default=128)
        parser.add_argument('--depth', type=int, default=4)
        parser.add_argument('--heads', type=int, default=4)
        parser.add_argument('--attn_dropout', default=0.1, type=float)
        parser.add_argument('--ff_dropout', default=0.1, type=float)
        parser.add_argument('--feature_redraw_interval',
                            type=int, default=1000)
        parser.add_argument(
            '--no_projection', default=False, action='store_true')
        return parser


class RelativePerformerModel(PerfomerBase):
    def __init__(self, dim, depth, heads, pos_scales, content_rel_attn=False,
                 pos_dims=1, attn_dropout=0., ff_dropout=0., max_pos=32,
                 feature_redraw_interval=100, no_projection=False, **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.save_hyperparameters()
        self.positional_embedding = LearnableSinusoidEncoding(
            pos_scales*2, max_timescale_init=max_pos*2)
        self.performer = RelativePerformer(
            dim,
            depth,
            heads,
            pos_dims=pos_dims,
            pos_scales=pos_scales,
            content_rel_attn=content_rel_attn,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            feature_redraw_interval=feature_redraw_interval,
            no_projection=no_projection
        )

    def forward(self, x, positions=None):
        embedding = self.content_embedding(x)
        # If positions are provided we assume the input has already been
        # flattened
        if positions is None:
            embedding, positions = self._flatten_to_sequence(embedding)
        positions = self._compute_positional_embeddings(positions)
        # First element contains class prediction
        embedding, positions = self._add_class_query(embedding, positions)
        out = self.performer(embedding, positions)[:, 0]
        return self.output_layer(out)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument('--dim', type=int, default=128)
        parser.add_argument('--depth', type=int, default=4)
        parser.add_argument('--heads', type=int, default=4)
        parser.add_argument('--attn_dropout', default=0.1, type=float)
        parser.add_argument('--ff_dropout', default=0.1, type=float)
        parser.add_argument('--pos_scales', type=int, default=4)
        parser.add_argument('--feature_redraw_interval',
                            type=int, default=1000)
        parser.add_argument(
            '--no_projection', default=False, action='store_true')
        parser.add_argument(
            '--content_rel_attn', default=False, action='store_true',
            help='Relative positional attention conditional on content'
        )
        return parser


class ClippedRelativePerformerModel(PerfomerBase):
    def __init__(self, dim, depth, heads, max_pos=32, max_rel_dist=8, **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.save_hyperparameters()
        self.performer = ClippedRelativePerformer(
            dim,
            depth,
            heads,
            max_rel_dist=max_rel_dist
        )

    def forward(self, x):
        embedding = self.content_embedding(x)
        embedding = rearrange(embedding, 'b x y d -> b (x y) d')
        bs_embedding = embedding.shape[0]
        # Add learnt class query to input
        embedding = torch.cat(
            [
                embedding,
                self.class_query[None, None, :].expand(bs_embedding, -1, -1),
            ],
            axis=1
        )
        # Last element contains class prediction
        out = self.performer(embedding)[:, -1]
        return self.output_layer(out)

    @ staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--dim', type=int, default=128)
        parser.add_argument('--depth', type=int, default=4)
        parser.add_argument('--heads', type=int, default=4)
        parser.add_argument('--attn_dropout', default=0.1, type=float)
        parser.add_argument('--ff_dropout', default=0.1, type=float)
        parser.add_argument('--max_rel_dist', type=int, default=8)
        return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['Performer', 'RelativePerformer',
                                          'NoposPerformer', 'ClippedRelativePerformer'])
    parser.add_argument('dataset', choices=[
        'FashionMNIST', 'MNIST', 'CIFAR10'])
    parser.add_argument('--log_path', type=str, default='lightning_logs')
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--version', type=str, default=None)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument(
        '--embedding_type',
        default='linear',
        choices=['linear', 'MLP', 'lookup']
    )

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
    elif partial_args.model == 'ClippedRelativePerformer':
        parser = ClippedRelativePerformerModel.add_model_specific_args(parser)
        model_cls = ClippedRelativePerformerModel
    args = parser.parse_args()

    seed = pl.utilities.seed.seed_everything()
    print('Running with random seed: {}'.format(seed))
    args.seed = seed

    data_cls = getattr(datasets, args.dataset + 'DataModule')
    num_workers = 4
    # Handle incosistencies in DataModules: Some datasets accept batch_size,
    # some don't, some simply ignore it.
    if args.dataset == 'MNIST':
        if args.embedding_type == 'lookup':
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
            DATA_PATH.joinpath(args.dataset),
            num_workers=num_workers,
            train_transforms=transform,
            val_transforms=transform,
            test_transforms=transform
        )
    elif args.dataset == 'FashionMNIST':
        if args.embedding_type == 'lookup':
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
            DATA_PATH.joinpath(args.dataset),
            num_workers=num_workers,
            train_transforms=transform,
            val_transforms=transform,
            test_transforms=transform
        )
    elif args.dataset == 'CIFAR10':
        if args.embedding_type == 'lookup':
            transform = ToIntTensor
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        dataset = datasets.CIFAR10DataModule(
            DATA_PATH.joinpath(args.dataset),
            num_workers=num_workers,
            batch_size=args.batch_size,
            train_transforms=transform,
            val_transforms=transform,
            test_transforms=transform
        )

    dataset.prepare_data()
    dataset.setup()

    in_features, nx, ny = dataset.dims
    max_pos = max(nx, ny)
    max_pos = 32   # We resize all inputs to 32x32
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
        dirpath=Path(logger.log_dir).joinpath('checkpoints')
    )
    early_stopping_cb = pl.callbacks.EarlyStopping(
        monitor='val/acc', patience=20, mode='max', strict=True,
        verbose=1)
    lr_monitor = pl.callbacks.LearningRateMonitor('step')

    trainer = pl.Trainer(
        gpus=-1 if GPU_AVAILABLE else None,
        gradient_clip_val=0.5,  # Same as performer paper
        logger=logger,
        callbacks=[model_checkpoint_cb, early_stopping_cb, lr_monitor],
        accelerator='ddp'
    )

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
