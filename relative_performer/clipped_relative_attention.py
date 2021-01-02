"""Relative positional attention using clipped relative positional encodings."""
import math
from functools import partial
import torch
import torch.nn as nn
from einops import rearrange, repeat
from relative_performer.performer_pytorch import (
    default, exists, PreLayerNorm, ReZero, PreScaleNorm, Chunk,
    FeedForward, get_module_device, find_modules, gaussian_orthogonal_random_matrix,
    softmax_kernel)
from relative_performer.reversible import ReversibleSequence, SequentialSequence


class ClippedRelativeSelfAttention(nn.Module):
    def __init__(self, dim, causal=False, heads=8,
                 nb_features=None, feature_redraw_interval=1000,
                 generalized_attention=False, kernel_fn=nn.ReLU(),
                 qr_uniform_q=False, dropout=0., no_projection=False, max_rel_dist=4):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = dim // heads 
        inner_dim = dim_head * heads
        
        self.relative_attention = RelativeFastAttention(
            dim_head,
            nb_features,
            causal=causal,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            qr_uniform_q=qr_uniform_q,
            no_projection=no_projection
        )

        self.heads = heads

        self.to_q = nn.Linear(dim, inner_dim)
        self.rpe = nn.Parameter(torch.zeros((max_rel_dist+1, dim_head)))
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None, **kwargs):
        b, n, _, h = *x.shape, self.heads

        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(
            context_mask, mask) if not cross_attend else context_mask

        q, rpe, v = self.to_q(x), self.rpe, self.to_v(context)

        q, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, v))
        # v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        if exists(context_mask):
            global_mask = context_mask[:, None, :, None]
            v.masked_fill_(~global_mask, 0.)

        out = self.relative_attention(q, rpe, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


class RelativeFastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), qr_uniform_q = False, no_projection = False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling, qr_uniform_q = qr_uniform_q)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, rpe, v):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim = -1)
            rpe = torch.exp(rpe) if self.causal else k.softmax(dim = -2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix, device = device)
            q, rpe = map(create_kernel, (q, rpe))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            q = create_kernel(q, is_query = True)
            rpe = create_kernel(rpe.unsqueeze(0).unsqueeze(0), is_query = False).squeeze() 

        out = relative_attention(q, rpe, v)
        return out


def relative_attention(q, rpe, v):
    max_rel_dist = rpe.shape[0]-1
    v_sum = v.sum(dim=-2)
    max_dist_enc = rpe[-1]
    max_dist = rpe.shape[0] -1
    L = q.shape[-2]
    img_dim = int(math.sqrt(L))
    dim = q.shape[-1]
    batch_size = q.shape[0]
    heads = q.shape[1]

    x_diffs = torch.arange(-max_dist+1, max_dist).repeat(max_dist*2-1).expand(L, -1)
    y_diffs = torch.arange(-max_dist+1, max_dist).repeat_interleave(max_dist*2-1).expand(L, -1)
    x_pos = x_diffs+torch.cat([torch.arange(img_dim).repeat(img_dim).unsqueeze(0).T, torch.zeros((1,1), dtype=torch.long)])
    y_pos = y_diffs+torch.cat([torch.arange(img_dim).repeat_interleave(img_dim).unsqueeze(0).T, torch.zeros((1,1), dtype=torch.long)+img_dim])
    diffs = torch.abs(x_diffs) + torch.abs(y_diffs)
    valid = torch.logical_and(torch.logical_and(torch.logical_and(x_pos >=0, x_pos < img_dim), torch.logical_and(y_pos >=0, y_pos < img_dim)), diffs<max_dist)
    diffs[valid != True] = max_rel_dist

    q_dot_rpe = q @ (rpe - max_dist_enc).T
    q_idx = torch.arange(0,diffs.shape[0]).unsqueeze(1).expand_as(diffs)
    q_rel = q_dot_rpe[:,:,q_idx, diffs]
    q_rel[:,:,valid != True] = 0

    rel_window = (max_rel_dist-1)*img_dim+max_rel_dist-1
    v_padded = torch.zeros((batch_size, heads, rel_window*2 + L, v.shape[-1]))
    v_padded[:,:,rel_window:rel_window+L,:] = v
    v_unfolded = v_padded.unfold(2,rel_window*2+1,1).permute(0,1,2,4,3)
    indices = (torch.arange(0,max_rel_dist*2-1).repeat(max_rel_dist*2-1) + torch.arange(0,(max_rel_dist*2-1)*10, 10).repeat_interleave(max_rel_dist*2-1))
    # TODO: make sparse tensor
    q_rel_sparse = torch.zeros((batch_size, heads, L, rel_window*2+1))
    q_rel_sparse[:,:,:,indices] = q_rel

    q_max_dist = torch.einsum('...ij,j->...i', q, max_dist_enc)
    D_inv =  1. / torch.einsum('...ij,ij->...i', q, (rpe[diffs] - max_dist_enc).sum(dim=1) + max_dist_enc * L)
    out = torch.einsum('...i,...j->...ij', q_max_dist, v_sum) + torch.einsum('...ij,...ijk->...ik', q_rel_sparse, v_unfolded) 
    out = torch.einsum('...ij,...i->...ij', out, D_inv)
    return out


class ClippedRelativePerformer(nn.Module):
    def __init__(self, dim, depth, heads, max_rel_dist=8, causal = False, ff_mult = 4, nb_features = None, feature_redraw_interval = 1000, reversible = False, ff_chunks = 1, generalized_attention = False, kernel_fn = nn.ReLU(), qr_uniform_q = False, use_scalenorm = False, use_rezero = False, ff_glu = False, ff_dropout = 0., attn_dropout = 0., cross_attend = False, no_projection = False):
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
                    ClippedRelativeSelfAttention(
                        dim, causal=causal, heads=heads,
                        nb_features=nb_features,
                        generalized_attention=generalized_attention,
                        kernel_fn=kernel_fn,
                        qr_uniform_q=qr_uniform_q,
                        dropout=attn_dropout,
                        no_projection=no_projection,
                        max_rel_dist=max_rel_dist)),
                wrapper_fn(
                    Chunk(
                        ff_chunks,
                        FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu),
                        along_dim=1)
                )
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)
        # route_positions = ((True, False),) * depth * (2 if cross_attend else 1)
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'mask': route_attn}
        # positions_route_map = {'pos': route_positions}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        self.net = execute_type(layers, args_route={**attn_route_map, **context_route_map, 
        # **positions_route_map
        })

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

    def forward(self, x, **kwargs):
        self.check_redraw_projections()
        return self.net(x, **kwargs)

def _headwise_causal_numerator(q_prime, k_prime_t, v):
    results = []

    # Iterate over the attention heads to avoid allocating a very large tensor
    for head in range(q_prime.shape[1]):
        # Outer products- a sorta biggish tensor
        outer_prods = torch.einsum('bml,bld->blmd', k_prime_t[:, head], v[:, head])
        prefix_sums = outer_prods.cumsum(dim=1)

        query_prods = torch.einsum('blmd,blm->bld', prefix_sums, q_prime[:, head])
        results.append(query_prods.unsqueeze(1))

    return torch.cat(results, dim=1)

if __name__ == '__main__':
    model = ClippedRelativePerformer(
        dim=512, depth=1, heads=8, causal=False
    )
    x = torch.randn(1, 1024, 512)
    model(x)
