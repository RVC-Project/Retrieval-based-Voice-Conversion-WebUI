from __future__ import annotations
from math import pi, log
import warnings

warnings.filterwarnings(
    "ignore",
    message="`torch.cuda.amp.autocast.*is deprecated.*",
    category=FutureWarning,
)

import torch
from torch.nn import Module, ModuleList
from torch.cuda.amp import autocast
from torch import nn, einsum, broadcast_tensors, Tensor
from einops import rearrange, repeat
from typing import Literal

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def broadcat(tensors, dim=-1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim=dim)

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    (x1, x2) = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast(enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    dtype = t.dtype
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    (t_left, t, t_right) = (t[..., :start_index], t[..., start_index:end_index], t[..., end_index:])
    t = t * freqs.cos() * scale + rotate_half(t) * freqs.sin() * scale
    if t.device.type == 'privateuseone':
        # DirectML rejects concatenation when one of the slices has a zero
        # length.  Rotary embeddings normally cover the complete head, so both
        # edge slices are empty; omitting them is mathematically identical.
        parts = tuple(part for part in (t_left, t, t_right) if part.shape[-1] > 0)
        out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
    else:
        out = torch.cat((t_left, t, t_right), dim=-1)
    return out.type(dtype)

def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')
    rotations = repeat(rotations, '... n -> ... (n r)', r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)

class RotaryEmbedding(Module):

    def __init__(self, dim, custom_freqs=None, freqs_for='lang', theta=10000, max_freq=10, num_freqs=1, learned_freq=False, use_xpos=False, xpos_scale_base=512, interpolate_factor=1.0, theta_rescale_factor=1.0, seq_before_head_dim=False, cache_if_possible=True):
        super().__init__()
        theta *= theta_rescale_factor ** (dim / (dim - 2))
        self.freqs_for = freqs_for
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        self.cache_if_possible = cache_if_possible
        self.tmp_store('cached_freqs', None)
        self.tmp_store('cached_scales', None)
        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)
        self.learned_freq = learned_freq
        self.tmp_store('dummy', torch.tensor(0))
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2
        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor
        self.use_xpos = use_xpos
        if not use_xpos:
            self.tmp_store('scale', None)
            return
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store('scale', scale)
        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent=False)

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, scale=None):
        seq_dim = default(seq_dim, self.default_seq_dim)
        assert not self.use_xpos or exists(scale), 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'
        (device, dtype, seq_len) = (t.device, t.dtype, t.shape[seq_dim])
        seq = self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset)
        freqs = self.forward(seq, seq_len=seq_len, offset=offset)
        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
        return apply_rotary_emb(freqs, t, scale=default(scale, 1.0), seq_dim=seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim=None, offset=0):
        (dtype, device, seq_dim) = (q.dtype, q.device, default(seq_dim, self.default_seq_dim))
        (q_len, k_len) = (q.shape[seq_dim], k.shape[seq_dim])
        assert q_len <= k_len
        q_scale = k_scale = 1.0
        if self.use_xpos:
            seq = self.get_seq_pos(k_len, dtype=dtype, device=device)
            q_scale = self.get_scale(seq[-q_len:]).type(dtype)
            k_scale = self.get_scale(seq).type(dtype)
        rotated_q = self.rotate_queries_or_keys(q, seq_dim=seq_dim, scale=q_scale, offset=k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim=seq_dim, scale=k_scale ** (-1))
        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)
        return (rotated_q, rotated_k)

    def rotate_queries_and_keys(self, q, k, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)
        assert self.use_xpos
        (device, dtype, seq_len) = (q.device, q.dtype, q.shape[seq_dim])
        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)
        freqs = self.forward(seq, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)
        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')
        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale ** (-1), seq_dim=seq_dim)
        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)
        return (rotated_q, rotated_k)

    def get_scale(self, t, seq_len=None, offset=0):
        assert self.use_xpos
        should_cache = self.cache_if_possible and exists(seq_len)
        if should_cache and exists(self.cached_scales) and (seq_len + offset <= self.cached_scales.shape[0]):
            return self.cached_scales[offset:offset + seq_len]
        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = torch.cat((scale, scale), dim=-1)
        if should_cache:
            self.tmp_store('cached_scales', scale)
        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []
        for (ind, dim) in enumerate(dims):
            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)
            freqs = self.forward(pos, seq_len=dim)
            all_axis = [None] * len(dims)
            all_axis[ind] = Colon
            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])
        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    @autocast(enabled=False)
    def forward(self, t, seq_len=None, offset=0):
        should_cache = self.cache_if_possible and (not self.learned_freq) and exists(seq_len) and (self.freqs_for != 'pixel')
        if should_cache and exists(self.cached_freqs) and (offset + seq_len <= self.cached_freqs.shape[0]):
            return self.cached_freqs[offset:offset + seq_len].detach()
        freqs = self.freqs
        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        if should_cache:
            self.tmp_store('cached_freqs', freqs.detach())
        return freqs
