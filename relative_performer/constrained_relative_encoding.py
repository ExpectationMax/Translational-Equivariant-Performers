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
        # Stack + reshape instead of concat, this way we get features of the
        # form [sin(w_1 * x), cos(w_1 * x), sin(w_2 * x), cos(w_2 * x)] instead
        # of [sin(w_1 * x), sin(w_2 *x), cos(w_1 * x), cos(w_2 * x)].
        emb = torch.stack((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb.view(*emb.shape[:-2], -1)


class ConstrainedLinear(nn.Module):
    """A linear layer with constraints for positional dimensions.

    This linear layer behaves the same as a regular linear layer for dimensions
    of the input associated with content of input elements, yet applies
    a constrained linear operation on the dimensions associated with positional
    embeddings.
    access to is purely relative.
    """

    def __init__(self, in_features, out_features, pos_scales, heads,
                 content_rel_attn=False,
                 bias=True):
        """Initialize ConstrainedLinear layer.

        Args:
            dim_in: Dimensionality of the input elements.
            dim_out: Dimensionality of the output (excluding the dimensions
                corresponding to the positional encoding).
            n_pos_scales: Number of sin/cos pairs with same lengthscale
                in the positional encoding.
            heads: Number of heads.
            bias: Include a bias.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.pos_scales = pos_scales
        self.heads = heads
        self.content_rel_attn = content_rel_attn
        # Number of features per head
        positional_features_head = 2*pos_scales
        self.content_linear = nn.Linear(in_features, out_features)
        if self.content_rel_attn:
            self.content_to_rel_matrix = nn.Linear(in_features, 2*heads*pos_scales)

        self.alpha = nn.Parameter(
            torch.Tensor(pos_scales*heads))
        self.beta = nn.Parameter(
            torch.Tensor(pos_scales*heads))
        self.register_buffer(
            'offdiag_matrix', torch.Tensor([[0., 1.], [-1., 0.]]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.alpha)
        init.normal_(self.beta)

    def _build_positional_projection_matrix(self):
        """Build projection matrices for positional encodings.

        Returns:
            Tensor with shape with shape [heads, pos_scales, 2, 2].
        """
        matrix = rearrange(
            torch.stack(
                [self.alpha, self.beta, -self.beta, self.alpha], dim=-1),
            '(h s) (b1 b2) -> h s b1 b2',
            h=self.heads,
            s=self.pos_scales,
            b1=2, b2=2
        )
        return matrix

    def _build_conditional_projection_matrix(self, input):
        parameters = rearrange(
            self.content_to_rel_matrix(input),
            'b n (h s d) -> b h n s d', d=2, h=self.heads, s=self.pos_scales)
        alpha, beta = torch.split(parameters, 2, dim=-1)
        matrix = torch.cat([alpha, -beta, beta, alpha], dim=-1)
        return matrix

    def forward(self, input: torch.Tensor, pos_encodings: torch.Tensor):
        bs = input.shape[0]
        content_based = rearrange(
            self.content_linear(input),
            'b n (h d) -> b h n d',
            h=self.heads
        )
        pos_encodings = rearrange(
            pos_encodings, 'b n (s d) -> b 1 s n d', s=self.pos_scales, d=2)
        # Format batch_size, heads, scales, instances, 2
        position_based = pos_encodings.matmul(
            self._build_positional_projection_matrix())
        position_based = rearrange(position_based, 'b h s n d -> b h n (s d)')

        return torch.cat(
            [content_based, position_based.expand(bs, -1, -1, -1)],
            axis=-1)


class IdentityLinear(nn.Module):
    """A linear layer with identity for positional dimensions.

    This linear layer behaves the same as a regular linear layer for dimensions
    of the input associated with content of input elements, yet returns the
    unmodified positional embeddings.

    This constraint ensures that the position information the network has
    access to is purely relative.
    """

    def __init__(self, in_features, out_features, pos_scales, heads,
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
        self.pos_scales = pos_scales
        self.heads = heads
        # Number of features per head
        positional_features_head = 2*pos_scales
        self.content_linear = nn.Linear(in_features, out_features)

    def forward(self, input: torch.Tensor, pos_encodings: torch.Tensor):
        bs = input.shape[0]
        content_based = rearrange(
            self.content_linear(input),
            'b n (h d) -> b h n d',
            h=self.heads
        )
        position_based = repeat(
            pos_encodings, 'b n d -> b h n d', h=self.heads)
        return torch.cat(
            [content_based, position_based.expand(bs, -1, -1, -1)
             ], axis=-1)


class RelPosSelfAttention(nn.Module):
    def __init__(self, dim, causal=False, heads=8, pos_scales=4,
                 nb_features=None, feature_redraw_interval=1000,
                 generalized_attention=False, kernel_fn=nn.ReLU(),
                 qr_uniform_q=False, dropout=0., no_projection=False):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = dim // heads + 2*pos_scales
        inner_dim = dim
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

        self.to_q = ConstrainedLinear(
            dim, inner_dim, pos_scales, self.heads)
        self.to_k = IdentityLinear(
            dim, inner_dim, pos_scales, self.heads)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos, context=None, mask=None, context_mask=None, **kwargs):
        b, n, _, h = *x.shape, self.heads

        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(
            context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x, pos), self.to_k(context, pos), self.to_v(context)

        # q and k are already in the right format
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        if exists(context_mask):
            global_mask = context_mask[:, None, :, None]
            v.masked_fill_(~global_mask, 0.)

        out = self.fast_attention(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


class RelativePerformer(nn.Module):
    def __init__(self, dim, depth, heads, pos_dims=1, pos_scales=4, causal = False, ff_mult = 4, nb_features = None, feature_redraw_interval = 1000, reversible = False, ff_chunks = 1, generalized_attention = False, kernel_fn = nn.ReLU(), qr_uniform_q = False, use_scalenorm = False, use_rezero = False, ff_glu = False, ff_dropout = 0., attn_dropout = 0., cross_attend = False, no_projection = False):
        super().__init__()
        layers = nn.ModuleList([])

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _ in range(depth):
            layers.append(nn.ModuleList([
                wrapper_fn(
                    RelPosSelfAttention(
                        dim, causal=causal, heads=heads,
                        pos_scales=pos_dims*pos_scales,
                        nb_features=nb_features,
                        generalized_attention=generalized_attention,
                        kernel_fn=kernel_fn,
                        qr_uniform_q=qr_uniform_q,
                        dropout=attn_dropout,
                        no_projection=no_projection)),
                wrapper_fn(
                    Chunk(
                        ff_chunks,
                        FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu),
                        along_dim=1)
                )
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)
        route_positions = ((True, False),) * depth * (2 if cross_attend else 1)
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'mask': route_attn}
        positions_route_map = {'pos': route_positions}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        self.net = execute_type(layers, args_route={**attn_route_map, **context_route_map, **positions_route_map})

        # keeping track of when to redraw projections for all attention layers
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projection_matrices_(self):
        self.feature_redraw_interval = None

    def check_redraw_projections(self):
        if not self.training:
            return

        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(self)

            fast_attentions = find_modules(self, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x, positions, **kwargs):
        self.check_redraw_projections()
        return self.net(x, pos=positions, **kwargs)


if __name__ == '__main__':
    model = RelativePerformer(
        dim=512, depth=1, heads=8, pos_dims=1, pos_scales=4, causal=False
    )
    positions = torch.arange(1024, dtype=torch.float32)[None, :, None]
    pos_embedding = rearrange(
        LearnableSinusoidEncoding(4*2)(positions),
        'b n p d -> b n (p d)'
    )
    x = torch.randn(1, 1024, 512)
    model(x, pos_embedding)
