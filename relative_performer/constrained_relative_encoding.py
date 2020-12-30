"""Relative positional attention using constrained projections."""
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat
from relative_performer.performer_pytorch import (
    default, exists, FastAttention, PreLayerNorm, ReZero, PreScaleNorm, Chunk,
    FeedForward, get_module_device, find_modules)
from relative_performer.reversible import ReversibleSequence, SequentialSequence


class LearnableSinusoidEncoding(nn.Module):
    """Layer converts scalar input to Sinusoid Encoding with learnt scaling."""

    def __init__(self, dim, max_timescale_init=10000):
        """Initialize layer.

        Args:
            dim: Dimensionality of the sinusoid encoding, should be dividable
                by 2.
            max_timescale_init: Maximum time scale used during initialization.
        """
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1. / (
            max_timescale_init ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq = nn.Parameter(inv_freq, requires_grad=True)

    def forward(self, x):
        sinusoid_inp = torch.matmul(
            x[..., None], self.inv_freq[None, :])
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb


class ConstrainedLinear(nn.Module):
    """A linear layer with constraints for positional dimensions.

    This linear layer behaves the same as a regular linear layer for dimensions
    of the input associated with content of input elements, yet applies
    a constrained linear operation on the dimensions associated with positional
    embeddings.

    This constraint ensures that the position information the network has
    access to is purely relative.
    """

    def __init__(self, in_features, out_features, n_pos_lengthscales_in, heads,
                 bias=True):
        """Initialize ConstrainedLinear layer.

        Args:
            dim_in: Dimensionality of the input elements.
            dim_out: Dimensionality of the output (excluding the dimensions
                corresponding to the positional encoding).
            n_pos_lengthscales: Number of sin/cos pairs with same lengthscale
                in the positional encoding.
            heads: Number of heads.
            bias: Include a bias.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_pos_lengthscales_in = n_pos_lengthscales_in
        self.heads = heads
        self.positional_features = (
            n_pos_lengthscales_in*heads)

        self.content_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.alpha = nn.Parameter(
            torch.Tensor(self.positional_features))
        self.beta = nn.Parameter(
            torch.Tensor(self.positional_features))
        self.register_buffer(
            'offdiag_matrix', torch.Tensor([[0., 1.], [-1., 0.]]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(
                self.out_features+self.positional_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.content_weight, a=math.sqrt(5))
        init.normal_(self.alpha)
        init.normal_(self.beta)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.content_weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _build_positional_projection_matrix(self):
        """Build projection matrix for positional encodings.

        Returns:
            Block diagonal matrix with shape [heads, n_pos_lengthscales,
            n_pos_lengthscales].
        """
        device = self.alpha.device
        identity_matrix = torch.diag(torch.ones(2, device=device))
        matrices = (
            self.alpha[:, None] * identity_matrix[None, :]
            + self.beta[:, None] * self.offdiag_matrix[None, :]
        )
        return torch.stack([
            torch.block_diag(*matrices[i:i+(1*self.n_pos_lengthscales_in)])
            for i in range(
                0, self.positional_features, self.n_pos_lengthscales_in)
        ], axis=0)

    def forward(self, input: torch.Tensor, pos_encodings: torch.Tensor):
        content_based = F.linear(input, self.content_weight)
        content_based = rearrange(
            content_based, 'b n (h d) -> b h n d', h=self.heads)
        position_based = rearrange(pos_encodings, 'b n d -> b n 1 1 d')
        position_based = position_based.matmul(
            self._build_positional_projection_matrix())
        position_based = rearrange(position_based, 'b n h 1 d -> b h n d')
        return torch.cat([content_based, position_based], axis=-1) + self.bias


class IdentityLinear(nn.Module):
    """A linear layer with identity for positional dimensions.

    This linear layer behaves the same as a regular linear layer for dimensions
    of the input associated with content of input elements, yet returns the
    unmodified positional embeddings.

    This constraint ensures that the position information the network has
    access to is purely relative.
    """

    def __init__(self, in_features, out_features, n_pos_lengthscales_in, heads,
                 bias=True):
        """Initialize IdentityLinear layer.

        Args:
            dim_in: Dimensionality of the input elements.
            dim_out: Dimensionality of the output (excluding the dimensions
                corresponding to the positional encoding).
            n_pos_lengthscales: Number of sin/cos pairs with same lengthscale
                in the positional encoding.
            heads: Number of heads.
            bias: Include a bias.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_pos_lengthscales_in = n_pos_lengthscales_in
        self.heads = heads
        self.positional_features = (
            n_pos_lengthscales_in*heads)

        self.content_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(
                self.out_features+self.positional_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.content_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.content_weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, pos_encodings: torch.Tensor):
        content_based = F.linear(input, self.content_weight)
        content_based = rearrange(
            content_based, 'b n (h d) -> b h n d', h=self.heads)
        position_based = repeat(
            pos_encodings, 'b n d -> b h n d', h=self.heads)
        return torch.cat([content_based, position_based], axis=-1) + self.bias


class RelPosSelfAttention(nn.Module):
    def __init__(self, dim, causal=False, heads=8, dim_head=64,
                 pos_lengthscales=4, nb_features=None,
                 feature_redraw_interval=1000, generalized_attention=False,
                 kernel_fn=nn.ReLU(), qr_uniform_q=False, dropout=0.,
                 no_projection=False):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(
            dim_head,
            nb_features,
            causal=causal,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            qr_uniform_q=qr_uniform_q,
            no_projection=no_projection
        )

        self.heads = heads

        self.to_q = ConstrainedLinear(dim, inner_dim)
        self.to_k = IdentityLinear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos, context=None, mask=None, context_mask=None, **kwargs):
        b, n, _, h = *x.shape, self.heads

        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(
            context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        # q and k are already in the right format
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        if exists(context_mask):
            global_mask = context_mask[:, None, :, None]
            v.masked_fill_(~global_mask, 0.)

        out = self.fast_attention(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)
