import math
import random
import sys
import types
import warnings
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import yaml


LEGACY_STEMS_4 = ["drums", "bass", "other", "vocals"]
LEGACY_STEMS_2 = ["vocals", "non_vocals"]


def center_trim(tensor, reference):
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    if delta < 0:
        raise ValueError(f"tensor must be larger than reference. Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
    return tensor


class BLSTM(nn.Module):
    def __init__(self, dim, layers=1, max_steps=None, skip=False):
        super().__init__()
        self.max_steps = max_steps
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)
        self.skip = skip

    def forward(self, x):
        y = x
        framed = False
        if self.max_steps is not None and x.shape[-1] > self.max_steps:
            batch, channels, length = x.shape
            width = self.max_steps
            stride = width // 2
            frames = x.unfold(-1, width, stride)
            nframes = frames.shape[2]
            framed = True
            x = frames.permute(0, 2, 1, 3).reshape(-1, channels, width)
        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        if framed:
            out = []
            frames = x.reshape(batch, -1, channels, width)
            limit = stride // 2
            for index in range(nframes):
                if index == 0:
                    out.append(frames[:, index, :, :-limit])
                elif index == nframes - 1:
                    out.append(frames[:, index, :, limit:])
                else:
                    out.append(frames[:, index, :, limit:-limit])
            x = torch.cat(out, -1)[..., :length]
        return x + y if self.skip else x


def _rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            std = sub.weight.std().detach()
            scale = (std / reference) ** 0.5
            sub.weight.data /= scale
            if sub.bias is not None:
                sub.bias.data /= scale


def _resample_x2(x):
    return F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)


def _downsample_x2(x, length):
    x = F.interpolate(x, size=length, mode="linear", align_corners=False)
    return x


class LegacyDemucs(nn.Module):
    def __init__(
        self,
        sources=4,
        audio_channels=2,
        channels=64,
        depth=6,
        rewrite=True,
        glu=True,
        rescale=0.1,
        resample=True,
        upsample=None,
        kernel_size=8,
        stride=4,
        growth=2.0,
        lstm_layers=2,
        context=3,
        normalize=False,
        samplerate=44100,
        segment_length=4 * 10 * 44100,
        **_,
    ):
        super().__init__()
        if upsample is not None:
            resample = bool(upsample)
        self.audio_channels = audio_channels
        self.sources = _normalize_sources(sources)
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.resample = resample
        self.channels = channels
        self.normalize = normalize
        self.samplerate = samplerate
        self.segment_length = segment_length

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1
        in_channels = audio_channels
        for index in range(depth):
            encode = [nn.Conv1d(in_channels, channels, kernel_size, stride), nn.ReLU()]
            if rewrite:
                encode += [nn.Conv1d(channels, ch_scale * channels, 1), activation]
            self.encoder.append(nn.Sequential(*encode))

            out_channels = in_channels if index > 0 else len(self.sources) * audio_channels
            decode = []
            if rewrite:
                decode += [nn.Conv1d(channels, ch_scale * channels, context), activation]
            decode += [nn.ConvTranspose1d(channels, out_channels, kernel_size, stride)]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels = int(growth * channels)

        if lstm_layers:
            self.lstm = BLSTM(in_channels, lstm_layers)
        else:
            self.lstm = None

        if rescale:
            _rescale_module(self, reference=rescale)

    def valid_length(self, length):
        if self.resample:
            length *= 2
        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            length += self.context - 1
        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        if self.resample:
            length = math.ceil(length / 2)
        return int(length)

    def forward(self, mix):
        length = mix.shape[-1]
        x = mix
        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            mean = mono.mean(dim=-1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
        else:
            mean = 0
            std = 1

        x = (x - mean) / (1e-5 + std)
        if self.resample:
            x = _resample_x2(x)

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)
        if self.lstm:
            x = self.lstm(x)
        for decode in self.decoder:
            skip = center_trim(saved.pop(-1), x)
            x = x + skip
            x = decode(x)

        if self.resample:
            x = _downsample_x2(x, length)
        x = x * std + mean
        return x.view(x.size(0), len(self.sources), self.audio_channels, x.size(-1))


class LegacyV3Demucs(nn.Module):
    def __init__(
        self,
        sources,
        audio_channels=2,
        channels=64,
        growth=2.0,
        depth=6,
        rewrite=True,
        lstm_layers=0,
        kernel_size=8,
        stride=4,
        context=1,
        gelu=True,
        glu=True,
        norm_starts=4,
        norm_groups=4,
        dconv_mode=1,
        dconv_depth=2,
        dconv_comp=4,
        dconv_attn=4,
        dconv_lstm=4,
        dconv_init=1e-4,
        normalize=True,
        resample=True,
        rescale=0.1,
        samplerate=44100,
        segment=4 * 10,
        **_,
    ):
        super().__init__()
        self.audio_channels = audio_channels
        self.sources = _normalize_sources(sources)
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.resample = resample
        self.channels = channels
        self.normalize = normalize
        self.samplerate = samplerate
        self.segment = segment
        self.segment_length = int(float(segment) * samplerate)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        activation = nn.GLU(dim=1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1
        act2 = nn.GELU if gelu else nn.ReLU
        in_channels = audio_channels
        for index in range(depth):
            norm_fn = (lambda d: nn.GroupNorm(norm_groups, d)) if index >= norm_starts else (lambda d: nn.Identity())
            attn = index >= dconv_attn
            lstm = index >= dconv_lstm
            encode = [
                nn.Conv1d(in_channels, channels, kernel_size, stride),
                norm_fn(channels),
                act2(),
            ]
            if dconv_mode & 1:
                encode.append(
                    LegacyDConv(channels, depth=dconv_depth, init=dconv_init, compress=dconv_comp, attn=attn, lstm=lstm)
                )
            if rewrite:
                encode += [nn.Conv1d(channels, ch_scale * channels, 1), norm_fn(ch_scale * channels), activation]
            self.encoder.append(nn.Sequential(*encode))

            out_channels = in_channels if index > 0 else len(self.sources) * audio_channels
            decode = []
            if rewrite:
                decode += [
                    nn.Conv1d(channels, ch_scale * channels, 2 * context + 1, padding=context),
                    norm_fn(ch_scale * channels),
                    activation,
                ]
            if dconv_mode & 2:
                decode.append(
                    LegacyDConv(channels, depth=dconv_depth, init=dconv_init, compress=dconv_comp, attn=attn, lstm=lstm)
                )
            decode.append(nn.ConvTranspose1d(channels, out_channels, kernel_size, stride))
            if index > 0:
                decode += [norm_fn(out_channels), act2()]
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels = int(growth * channels)

        self.lstm = BLSTM(in_channels, lstm_layers) if lstm_layers else None
        if rescale:
            _rescale_module(self, reference=rescale)

    def valid_length(self, length):
        if self.resample:
            length *= 2
        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        if self.resample:
            length = math.ceil(length / 2)
        return int(length)

    def forward(self, mix):
        x = mix
        length = x.shape[-1]
        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            mean = mono.mean(dim=-1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            x = (x - mean) / (1e-5 + std)
        else:
            mean = 0
            std = 1

        delta = self.valid_length(length) - length
        x = F.pad(x, (delta // 2, delta - delta // 2))
        if self.resample:
            x = _resample_x2(x)

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)
        if self.lstm:
            x = self.lstm(x)
        for decode in self.decoder:
            skip = center_trim(saved.pop(-1), x)
            x = decode(x + skip)
        if self.resample:
            x = _downsample_x2(x, length + delta)
        x = center_trim(x * std + mean, length)
        return x.view(x.size(0), len(self.sources), self.audio_channels, x.size(-1))


class LegacyLayerScale(nn.Module):
    def __init__(self, channels, init=0):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        return self.scale[:, None] * x


class LegacyLocalState(nn.Module):
    def __init__(self, channels, heads=4, nfreqs=0, ndecay=4):
        super().__init__()
        if channels % heads:
            raise ValueError("legacy local attention channels must be divisible by heads")
        self.heads = heads
        self.nfreqs = nfreqs
        self.ndecay = ndecay
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        if nfreqs:
            self.query_freqs = nn.Conv1d(channels, heads * nfreqs, 1)
        if ndecay:
            self.query_decay = nn.Conv1d(channels, heads * ndecay, 1)
            self.query_decay.weight.data *= 0.01
            self.query_decay.bias.data[:] = -2
        self.proj = nn.Conv1d(channels + heads * nfreqs, channels, 1)

    def forward(self, x):
        batch, channels, time = x.shape
        heads = self.heads
        indexes = torch.arange(time, device=x.device, dtype=x.dtype)
        delta = indexes[:, None] - indexes[None, :]

        queries = self.query(x).view(batch, heads, -1, time)
        keys = self.key(x).view(batch, heads, -1, time)
        dots = torch.einsum("bhct,bhcs->bhts", keys, queries)
        dots /= keys.shape[2] ** 0.5
        if self.nfreqs:
            periods = torch.arange(1, self.nfreqs + 1, device=x.device, dtype=x.dtype)
            freq_kernel = torch.cos(2 * math.pi * delta / periods.view(-1, 1, 1))
            freq_q = self.query_freqs(x).view(batch, heads, -1, time) / self.nfreqs**0.5
            dots += torch.einsum("fts,bhfs->bhts", freq_kernel, freq_q)
        if self.ndecay:
            decays = torch.arange(1, self.ndecay + 1, device=x.device, dtype=x.dtype)
            decay_q = torch.sigmoid(self.query_decay(x).view(batch, heads, -1, time)) / 2
            decay_kernel = -decays.view(-1, 1, 1) * delta.abs() / self.ndecay**0.5
            dots += torch.einsum("fts,bhfs->bhts", decay_kernel, decay_q)

        dots.masked_fill_(torch.eye(time, device=dots.device, dtype=torch.bool), -100)
        weights = torch.softmax(dots, dim=2)
        content = self.content(x).view(batch, heads, -1, time)
        result = torch.einsum("bhts,bhct->bhcs", weights, content)
        if self.nfreqs:
            time_sig = torch.einsum("bhts,fts->bhfs", weights, freq_kernel)
            result = torch.cat([result, time_sig], 2)
        result = result.reshape(batch, -1, time)
        return x + self.proj(result)


class LegacyDConv(nn.Module):
    def __init__(
        self,
        channels,
        compress=4,
        depth=2,
        init=1e-4,
        norm=True,
        attn=False,
        heads=4,
        ndecay=4,
        lstm=False,
        gelu=True,
        kernel=3,
        **_,
    ):
        super().__init__()
        norm_fn = (lambda d: nn.GroupNorm(1, d)) if norm else (lambda d: nn.Identity())
        act = nn.GELU if gelu else nn.ReLU
        hidden = int(channels / compress)
        self.layers = nn.ModuleList()
        for index in range(abs(depth)):
            dilation = 2**index if depth > 0 else 1
            padding = dilation * (kernel // 2)
            mods = [
                nn.Conv1d(channels, hidden, kernel, dilation=dilation, padding=padding),
                norm_fn(hidden),
                act(),
                nn.Conv1d(hidden, 2 * channels, 1),
                norm_fn(2 * channels),
                nn.GLU(1),
                LegacyLayerScale(channels, init),
            ]
            if attn:
                mods.insert(3, LegacyLocalState(hidden, heads=heads, ndecay=ndecay))
            if lstm:
                mods.insert(3, BLSTM(hidden, layers=2, max_steps=200, skip=True))
            self.layers.append(nn.Sequential(*mods))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class LegacyHEncLayer(nn.Module):
    def __init__(
        self,
        chin,
        chout,
        kernel_size=8,
        stride=4,
        norm_groups=1,
        empty=False,
        freq=True,
        dconv=True,
        norm=True,
        context=0,
        dconv_kw=None,
        pad=True,
        rewrite=True,
    ):
        super().__init__()
        dconv_kw = dconv_kw or {}
        norm_fn = (lambda d: nn.GroupNorm(norm_groups, d)) if norm else (lambda d: nn.Identity())
        pad = kernel_size // 4 if pad else 0
        klass = nn.Conv2d if freq else nn.Conv1d
        self.freq, self.kernel_size, self.stride, self.empty, self.norm, self.pad = freq, kernel_size, stride, empty, norm, pad
        if freq:
            kernel_size, stride, pad = [kernel_size, 1], [stride, 1], [pad, 0]
        self.conv = klass(chin, chout, kernel_size, stride, pad)
        if empty:
            return
        self.norm1 = norm_fn(chout)
        self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context) if rewrite else None
        if rewrite:
            self.norm2 = norm_fn(2 * chout)
        self.dconv = LegacyDConv(chout, **dconv_kw) if dconv else None

    def forward(self, x, inject=None):
        if not self.freq and x.dim() == 4:
            batch, _, _, time = x.shape
            x = x.view(batch, -1, time)
        if not self.freq and x.shape[-1] % self.stride:
            x = F.pad(x, (0, self.stride - x.shape[-1] % self.stride))
        y = self.conv(x)
        if self.empty:
            return y
        if inject is not None:
            if inject.dim() == 3 and y.dim() == 4:
                inject = inject[:, :, None]
            y = y + inject
        y = F.gelu(self.norm1(y))
        if self.dconv:
            if self.freq:
                batch, channels, freqs, time = y.shape
                y = (
                    self.dconv(y.permute(0, 2, 1, 3).reshape(-1, channels, time))
                    .view(batch, freqs, channels, time)
                    .transpose(1, 2)
                )
            else:
                y = self.dconv(y)
        return F.glu(self.norm2(self.rewrite(y)), dim=1) if self.rewrite else y


class LegacyHDecLayer(nn.Module):
    def __init__(
        self,
        chin,
        chout,
        last=False,
        kernel_size=8,
        stride=4,
        norm_groups=1,
        empty=False,
        freq=True,
        dconv=True,
        norm=True,
        context=1,
        dconv_kw=None,
        pad=True,
        context_freq=True,
        rewrite=True,
    ):
        super().__init__()
        dconv_kw = dconv_kw or {}
        norm_fn = (lambda d: nn.GroupNorm(norm_groups, d)) if norm else (lambda d: nn.Identity())
        self.pad = kernel_size // 4 if pad else 0
        self.last, self.freq, self.chin, self.empty, self.stride = last, freq, chin, empty, stride
        self.kernel_size, self.norm, self.context_freq = kernel_size, norm, context_freq
        klass, klass_tr = (nn.Conv2d, nn.ConvTranspose2d) if freq else (nn.Conv1d, nn.ConvTranspose1d)
        k, s = ([kernel_size, 1], [stride, 1]) if freq else (kernel_size, stride)
        self.conv_tr = klass_tr(chin, chout, k, s)
        self.norm2 = norm_fn(chout)
        if empty:
            return
        self.rewrite = None
        if rewrite:
            if context_freq:
                self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
            else:
                self.rewrite = klass(chin, 2 * chin, [1, 1 + 2 * context], 1, [0, context])
            self.norm1 = norm_fn(2 * chin)
        self.dconv = LegacyDConv(chin, **dconv_kw) if dconv else None

    def forward(self, x, skip, length):
        if self.freq and x.dim() == 3:
            batch, _, time = x.shape
            x = x.view(batch, self.chin, -1, time)
        if self.empty:
            y = x
        else:
            x = x + skip
            y = F.glu(self.norm1(self.rewrite(x)), dim=1) if self.rewrite else x
            if self.dconv:
                if self.freq:
                    batch, channels, freqs, time = y.shape
                    y = (
                        self.dconv(y.permute(0, 2, 1, 3).reshape(-1, channels, time))
                        .view(batch, freqs, channels, time)
                        .transpose(1, 2)
                    )
                else:
                    y = self.dconv(y)
        z = self.norm2(self.conv_tr(y))
        if self.freq and self.pad:
            z = z[..., self.pad : -self.pad, :]
        elif not self.freq:
            z = z[..., self.pad : self.pad + length]
            assert z.shape[-1] == length
        return (z if self.last else F.gelu(z)), y


class LegacyMultiWrap(nn.Module):
    def __init__(self, layer, split_ratios):
        from copy import deepcopy

        super().__init__()
        self.split_ratios = split_ratios
        self.conv = isinstance(layer, LegacyHEncLayer)

        def new_layer():
            cloned = deepcopy(layer)
            if self.conv:
                cloned.conv.padding = (0, 0)
            else:
                cloned.pad = False
            for module in cloned.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            return cloned

        self.layers = nn.ModuleList([new_layer() for _ in range(len(split_ratios) + 1)])

    def forward(self, x, skip=None, length=None):
        _, _, freqs, _ = x.shape
        start, outs = 0, []
        for ratio, layer in zip(list(self.split_ratios) + [1], self.layers):
            if self.conv:
                pad = layer.kernel_size // 4
                if ratio == 1:
                    limit = freqs
                else:
                    limit = int(round(freqs * ratio))
                    segment_length = limit - start + (pad if start == 0 else 0)
                    frames = round((segment_length - layer.kernel_size) / layer.stride + 1)
                    limit = start + (frames - 1) * layer.stride + layer.kernel_size - (pad if start == 0 else 0)
                y = x[:, :, start:limit, :]
                if start == 0:
                    y = F.pad(y, (0, 0, pad, 0))
                if ratio == 1:
                    y = F.pad(y, (0, 0, 0, pad))
                outs.append(layer(y))
                start = limit - layer.kernel_size + layer.stride
            else:
                limit = freqs if ratio == 1 else int(round(freqs * ratio))
                last = layer.last
                layer.last = True
                out, _ = layer(x[:, :, start:limit], skip[:, :, start:limit], length)
                if outs:
                    bias = layer.conv_tr.bias.view(1, -1, 1, 1)
                    outs[-1][:, :, -layer.stride :] += out[:, :, : layer.stride] - bias
                    out = out[:, :, layer.stride :]
                if ratio == 1:
                    out = out[:, :, : -layer.stride // 2, :]
                if start == 0:
                    out = out[:, :, layer.stride // 2 :, :]
                outs.append(out)
                layer.last = last
                start = limit
        out = torch.cat(outs, dim=2)
        return out if self.conv else (out if last else F.gelu(out), None)


class LegacyScaledEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, scale=10.0, smooth=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            weight = weight / torch.arange(1, num_embeddings + 1).to(weight).sqrt()[:, None]
            self.embedding.weight.data[:] = weight
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        return self.embedding(x) * self.scale


def _pad1d(x, paddings, mode="constant", value=0.0):
    x0 = x
    length = x.shape[-1]
    left, right = paddings
    if mode == "reflect":
        max_pad = max(left, right)
        if length <= max_pad:
            extra = max_pad - length + 1
            extra_right = min(right, extra)
            extra_left = extra - extra_right
            paddings = (left - extra_left, right - extra_right)
            x = F.pad(x, (extra_left, extra_right))
    out = F.pad(x, paddings, mode, value)
    assert out.shape[-1] == length + left + right
    assert (out[..., left : left + length] == x0).all()
    return out


def _spectro(x, n_fft=512, hop_length=None, pad=0):
    *other, length = x.shape
    z = torch.stft(
        x.reshape(-1, length),
        n_fft * (1 + pad),
        hop_length or n_fft // 4,
        window=torch.hann_window(n_fft).to(x),
        win_length=n_fft,
        normalized=True,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )
    return z.view(*other, z.shape[-2], z.shape[-1])


def _ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    win_length = n_fft // (1 + pad)
    x = torch.istft(
        z.reshape(-1, freqs, frames),
        n_fft,
        hop_length,
        window=torch.hann_window(win_length).to(z.real),
        win_length=win_length,
        normalized=True,
        length=length,
        center=True,
    )
    return x.view(*other, x.shape[-1])


class LegacyHDemucs(nn.Module):
    def __init__(
        self,
        sources,
        audio_channels=2,
        channels=48,
        channels_time=None,
        growth=2,
        nfft=4096,
        wiener_iters=0,
        end_iters=0,
        wiener_residual=False,
        cac=True,
        depth=6,
        rewrite=True,
        hybrid=True,
        hybrid_old=False,
        multi_freqs=None,
        multi_freqs_depth=2,
        freq_emb=0.2,
        emb_scale=10,
        emb_smooth=True,
        kernel_size=8,
        time_stride=2,
        stride=4,
        context=1,
        context_enc=0,
        norm_starts=4,
        norm_groups=4,
        dconv_mode=1,
        dconv_depth=2,
        dconv_comp=4,
        dconv_attn=4,
        dconv_lstm=4,
        dconv_init=1e-4,
        rescale=0.1,
        samplerate=44100,
        segment=4 * 10,
        **_,
    ):
        super().__init__()
        if wiener_iters != 0 or end_iters != 0 or not cac:
            raise ValueError("legacy HDemucs loader supports only CaC checkpoints without Wiener filtering")
        self.cac = cac
        self.wiener_residual = wiener_residual
        self.audio_channels = audio_channels
        self.sources = list(sources)
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.channels = channels
        self.samplerate = samplerate
        self.segment_length = int(float(segment) * samplerate)
        self.segment = segment
        self.nfft = nfft
        self.hop_length = nfft // 4
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None
        self.hybrid = hybrid
        self.hybrid_old = hybrid_old

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        if hybrid:
            self.tencoder = nn.ModuleList()
            self.tdecoder = nn.ModuleList()

        chin = audio_channels
        chin_z = chin * 2 if cac else chin
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2
        for index in range(depth):
            lstm = index >= dconv_lstm
            attn = index >= dconv_attn
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                ker = time_stride * 2
                stri = time_stride
            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True
            kw = {
                "kernel_size": ker,
                "stride": stri,
                "freq": freq,
                "pad": pad,
                "norm": norm,
                "rewrite": rewrite,
                "norm_groups": norm_groups,
                "dconv_kw": {
                    "lstm": lstm,
                    "attn": attn,
                    "depth": dconv_depth,
                    "compress": dconv_comp,
                    "init": dconv_init,
                    "gelu": True,
                },
            }
            kwt = dict(kw)
            kwt["freq"] = 0
            kwt["kernel_size"] = kernel_size
            kwt["stride"] = stride
            kwt["pad"] = True
            kw_dec = dict(kw)
            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            multi = bool(multi_freqs and index < multi_freqs_depth)
            if multi:
                kw_dec["context_freq"] = False

            enc = LegacyHEncLayer(chin_z, chout_z, dconv=dconv_mode & 1, context=context_enc, **kw)
            if multi:
                enc = LegacyMultiWrap(enc, multi_freqs)
            self.encoder.append(enc)
            if hybrid and freq:
                self.tencoder.append(
                    LegacyHEncLayer(chin, chout, dconv=dconv_mode & 1, context=context_enc, empty=last_freq, **kwt)
                )
            if index == 0:
                chin = audio_channels * len(self.sources)
                chin_z = chin * 2 if cac else chin
            dec = LegacyHDecLayer(chout_z, chin_z, dconv=dconv_mode & 2, last=index == 0, context=context, **kw_dec)
            if multi:
                dec = LegacyMultiWrap(dec, multi_freqs)
            self.decoder.insert(0, dec)
            if hybrid and freq:
                self.tdecoder.insert(
                    0,
                    LegacyHDecLayer(
                        chout, chin, dconv=dconv_mode & 2, empty=last_freq, last=index == 0, context=context, **kwt
                    ),
                )

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                freqs = 1 if freqs <= kernel_size else freqs // stride
            if index == 0 and freq_emb:
                self.freq_emb = LegacyScaledEmbedding(freqs, chin_z, smooth=emb_smooth, scale=emb_scale)
                self.freq_emb_scale = freq_emb

        if rescale:
            _rescale_module(self, reference=rescale)

    def valid_length(self, length):
        return length

    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        if self.hybrid:
            le = int(math.ceil(x.shape[-1] / hl))
            pad = hl // 2 * 3
            mode = "constant" if self.hybrid_old else "reflect"
            x = _pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode=mode)
        z = _spectro(x, nfft, hl)[..., :-1, :]
        if self.hybrid:
            z = z[..., 2 : 2 + le]
        return z

    def _ispec(self, z, length=None, scale=0):
        hl = self.hop_length // (4**scale)
        z = F.pad(z, (0, 0, 0, 1))
        if self.hybrid:
            z = F.pad(z, (2, 2))
            pad = hl // 2 * 3
            le = hl * int(math.ceil(length / hl)) + (0 if self.hybrid_old else 2 * pad)
            x = _ispectro(z, hl, length=le)
            x = x[..., :length] if self.hybrid_old else x[..., pad : pad + length]
            return x
        return _ispectro(z, hl, length)

    def _magnitude(self, z):
        if self.cac:
            batch, channels, freqs, time = z.shape
            return torch.view_as_real(z).permute(0, 1, 4, 2, 3).reshape(batch, channels * 2, freqs, time)
        return z.abs()

    def _mask(self, z, m):
        if self.cac:
            batch, sources, channels, freqs, time = m.shape
            out = m.view(batch, sources, -1, 2, freqs, time).permute(0, 1, 2, 4, 5, 3)
            return torch.view_as_complex(out.contiguous())
        raise ValueError("legacy HDemucs loader supports only CaC checkpoints")

    def forward(self, mix):
        length = mix.shape[-1]
        z = self._spec(mix)
        x = self._magnitude(z)
        batch, _, freqs, time = x.shape

        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        if self.hybrid:
            xt = mix
            meant = xt.mean(dim=(1, 2), keepdim=True)
            stdt = xt.std(dim=(1, 2), keepdim=True)
            xt = (xt - meant) / (1e-5 + stdt)

        saved, saved_t, lengths, lengths_t = [], [], [], []
        for index, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if self.hybrid and index < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[index]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if index == 0 and self.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb
            saved.append(x)

        x = torch.zeros_like(x)
        if self.hybrid:
            xt = torch.zeros_like(x)
        for index, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            if self.hybrid:
                offset = self.depth - len(self.tdecoder)
                if index >= offset:
                    tdec = self.tdecoder[index - offset]
                    length_t = lengths_t.pop(-1)
                    if tdec.empty:
                        pre = pre[:, :, 0]
                        xt, _ = tdec(pre, None, length_t)
                    else:
                        skip_t = saved_t.pop(-1)
                        xt, _ = tdec(xt, skip_t, length_t)

        sources = len(self.sources)
        x = x.view(batch, sources, -1, freqs, time)
        x = x * std[:, None] + mean[:, None]
        x = self._ispec(self._mask(z, x), length)
        if self.hybrid:
            xt = xt.view(batch, sources, -1, length)
            xt = xt * stdt[:, None] + meant[:, None]
            x = xt + x
        return x


EPS = 1e-8


def overlap_and_add(signal, frame_step):
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]
    subframe_length = math.gcd(frame_length, frame_step)
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)
    frame = torch.arange(0, output_subframes, device=signal.device).unfold(0, subframes_per_frame, subframe_step)
    frame = frame.long().contiguous().view(-1)
    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    return result.view(*outer_dimensions, -1)


class LegacyConvTasNet(nn.Module):
    def __init__(
        self,
        sources=None,
        N=256,
        L=20,
        B=256,
        H=512,
        P=3,
        X=8,
        R=4,
        C=4,
        audio_channels=2,
        norm_type="gLN",
        causal=False,
        mask_nonlinear="relu",
        samplerate=44100,
        segment_length=44100 * 2 * 4,
        **_,
    ):
        super().__init__()
        self.sources = _normalize_sources(C if sources is None else sources)
        self.C = len(self.sources)
        self.N, self.L, self.B, self.H, self.P, self.X, self.R = N, L, B, H, P, X, R
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        self.audio_channels = audio_channels
        self.samplerate = samplerate
        self.segment_length = segment_length
        self.encoder = Encoder(L, N, audio_channels)
        self.separator = TemporalConvNet(N, B, H, P, X, R, self.C, norm_type, causal, mask_nonlinear)
        self.decoder = Decoder(N, L, audio_channels)
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_normal_(parameter)

    def valid_length(self, length):
        return length

    def forward(self, mixture):
        mixture_w = self.encoder(mixture)
        est_mask = self.separator(mixture_w)
        est_source = self.decoder(mixture_w, est_mask)
        length = mixture.size(-1)
        delta = length - est_source.size(-1)
        return F.pad(est_source, (0, delta)) if delta >= 0 else est_source[..., :length]


class Encoder(nn.Module):
    def __init__(self, L, N, audio_channels):
        super().__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(audio_channels, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        return F.relu(self.conv1d_U(mixture))


class Decoder(nn.Module):
    def __init__(self, N, L, audio_channels):
        super().__init__()
        self.N, self.L = N, L
        self.audio_channels = audio_channels
        self.basis_signals = nn.Linear(N, audio_channels * L, bias=False)

    def forward(self, mixture_w, est_mask):
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask
        source_w = torch.transpose(source_w, 2, 3)
        est_source = self.basis_signals(source_w)
        batch, sources, frames, _ = est_source.size()
        est_source = est_source.view(batch, sources, frames, self.audio_channels, -1).transpose(2, 3).contiguous()
        return overlap_and_add(est_source, self.L // 2)


class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, norm_type="gLN", causal=False, mask_nonlinear="relu"):
        super().__init__()
        self.C = C
        self.mask_nonlinear = mask_nonlinear
        repeats = []
        for _ in range(R):
            blocks = []
            for index in range(X):
                dilation = 2**index
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                blocks.append(
                    TemporalBlock(B, H, P, stride=1, padding=padding, dilation=dilation, norm_type=norm_type, causal=causal)
                )
            repeats.append(nn.Sequential(*blocks))
        self.network = nn.Sequential(
            ChannelwiseLayerNorm(N),
            nn.Conv1d(N, B, 1, bias=False),
            nn.Sequential(*repeats),
            nn.Conv1d(B, C * N, 1, bias=False),
        )

    def forward(self, mixture_w):
        batch, channels, frames = mixture_w.size()
        score = self.network(mixture_w).view(batch, self.C, channels, frames)
        if self.mask_nonlinear == "softmax":
            return F.softmax(score, dim=1)
        if self.mask_nonlinear == "relu":
            return F.relu(score)
        raise ValueError("Unsupported mask non-linear function")


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, norm_type="gLN", causal=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.PReLU(),
            _choose_norm(norm_type, out_channels),
            DepthwiseSeparableConv(out_channels, in_channels, kernel_size, stride, padding, dilation, norm_type, causal),
        )

    def forward(self, x):
        return self.net(x) + x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, norm_type="gLN", causal=False):
        super().__init__()
        depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        layers = [depthwise_conv]
        if causal:
            layers.append(Chomp1d(padding))
        layers += [
            nn.PReLU(),
            _choose_norm(norm_type, in_channels),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class ChannelwiseLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        mean = torch.mean(y, dim=1, keepdim=True)
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)
        return self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta


class GlobalLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        var = torch.pow(y - mean, 2).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        return self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta


def _choose_norm(norm_type, channel_size):
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    if norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    if norm_type == "id":
        return nn.Identity()
    return nn.BatchNorm1d(channel_size)


class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length
        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)
        if isinstance(tensor, TensorChunk):
            self.tensor = tensor.tensor
            self.offset = offset + tensor.offset
        else:
            self.tensor = tensor
            self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0
        start = self.offset - delta // 2
        end = start + target_length
        correct_start = max(0, start)
        correct_end = min(total_length, end)
        pad_left = correct_start - start
        pad_right = end - correct_end
        out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
        assert out.shape[-1] == target_length
        return out


def tensor_chunk(tensor_or_chunk):
    return tensor_or_chunk if isinstance(tensor_or_chunk, TensorChunk) else TensorChunk(tensor_or_chunk)


class LegacyBagOfModels(nn.Module):
    def __init__(self, models, weights=None, segment=None):
        super().__init__()
        if not models:
            raise ValueError("legacy Demucs bag must contain at least one model")
        first = models[0]
        for model in models:
            if model.sources != first.sources:
                raise ValueError("all models in a legacy Demucs bag must have the same sources")
            if model.samplerate != first.samplerate:
                raise ValueError("all models in a legacy Demucs bag must have the same samplerate")
            if model.audio_channels != first.audio_channels:
                raise ValueError("all models in a legacy Demucs bag must have the same channel count")
            if segment is not None:
                model.segment_length = int(float(segment) * model.samplerate)
        self.sources = first.sources
        self.samplerate = first.samplerate
        self.audio_channels = first.audio_channels
        self.segment_length = first.segment_length
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [[1.0 for _ in self.sources] for _ in models]
        self.weights = weights

    def forward(self, x):
        raise NotImplementedError("use apply_legacy_model for legacy Demucs bags")


def apply_legacy_model(model, mix, shifts=0, split=True, overlap=0.25, transition_power=1.0, progress=False):
    if isinstance(model, LegacyBagOfModels):
        estimates = 0.0
        totals = [0.0] * len(model.sources)
        for sub_model, weights in zip(model.models, model.weights):
            out = apply_legacy_model(
                sub_model,
                mix,
                shifts=shifts,
                split=split,
                overlap=overlap,
                transition_power=transition_power,
                progress=progress,
            )
            for index, weight in enumerate(weights):
                out[index] *= weight
                totals[index] += weight
            estimates += out
        for index, total in enumerate(totals):
            estimates[index] /= total
        return estimates

    assert transition_power >= 1, "transition_power < 1 leads to unstable transitions"
    device = mix.device
    channels, length = mix.shape
    if split:
        out = torch.zeros(len(model.sources), channels, length, device=device)
        sum_weight = torch.zeros(length, device=device)
        segment = int(model.segment_length)
        stride = int((1 - overlap) * segment)
        weight = torch.cat(
            [
                torch.arange(1, segment // 2 + 1, device=device),
                torch.arange(segment - segment // 2, 0, -1, device=device),
            ]
        )
        weight = (weight / weight.max()) ** transition_power
        offsets = range(0, length, stride)
        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment)
            chunk_out = apply_legacy_model(model, chunk, shifts=shifts, split=False)
            chunk_length = chunk_out.shape[-1]
            out[..., offset : offset + segment] += weight[:chunk_length] * chunk_out
            sum_weight[offset : offset + segment] += weight[:chunk_length]
        out /= sum_weight
        return out
    if shifts:
        max_shift = int(0.5 * model.samplerate)
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0.0
        for _ in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            shifted_out = apply_legacy_model(model, shifted, shifts=0, split=False)
            out += shifted_out[..., max_shift - offset :]
        return out / shifts

    valid_length = model.valid_length(length)
    mix = tensor_chunk(mix)
    padded_mix = mix.padded(valid_length)
    with torch.no_grad():
        return center_trim(model(padded_mix.unsqueeze(0))[0], length)


class _PickleClassStub:
    pass


def _stub_class(module_name, class_name):
    return type(class_name, (_PickleClassStub,), {"__module__": module_name})


@contextmanager
def _legacy_pickle_modules():
    module_classes = {
        "demucs.model": {"Demucs": _stub_class("demucs.model", "Demucs")},
        "demucs.demucs": {"Demucs": _stub_class("demucs.demucs", "Demucs")},
        "demucs.tasnet": {"ConvTasNet": _stub_class("demucs.tasnet", "ConvTasNet")},
        "demucs.hdemucs": {"HDemucs": _stub_class("demucs.hdemucs", "HDemucs")},
        "demucs.htdemucs": {"HTDemucs": _stub_class("demucs.htdemucs", "HTDemucs")},
    }
    module_names = ["demucs", *module_classes.keys()]
    previous = {name: sys.modules.get(name) for name in module_names}
    package = sys.modules.setdefault("demucs", types.ModuleType("demucs"))
    package.__path__ = []
    try:
        for module_name, classes in module_classes.items():
            module = sys.modules.setdefault(module_name, types.ModuleType(module_name))
            for class_name, klass in classes.items():
                setattr(module, class_name, klass)
            setattr(package, module_name.split(".")[-1], module)
        yield module_classes
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _normalize_sources(sources):
    if isinstance(sources, int):
        if sources == 4:
            return LEGACY_STEMS_4.copy()
        if sources == 2:
            return LEGACY_STEMS_2.copy()
        return [f"source_{index}" for index in range(sources)]
    return list(sources)


def _resolve_klass(klass):
    module_name = getattr(klass, "__module__", "")
    class_name = getattr(klass, "__name__", "")
    if module_name == "demucs.model" and class_name == "Demucs":
        return LegacyDemucs
    if module_name == "demucs.demucs" and class_name == "Demucs":
        return LegacyV3Demucs
    if module_name == "demucs.tasnet" and class_name == "ConvTasNet":
        return LegacyConvTasNet
    if module_name == "demucs.hdemucs" and class_name == "HDemucs":
        return LegacyHDemucs
    if module_name == "demucs.htdemucs" and class_name == "HTDemucs":
        from .demucs4ht import HTDemucs

        return HTDemucs
    raise ValueError(f"Unsupported legacy Demucs checkpoint class: {module_name}.{class_name}")


def _load_raw_checkpoint(model_path):
    try:
        with _legacy_pickle_modules():
            return torch.load(model_path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        if exc.name == "diffq":
            raise ValueError("DiffQ quantized legacy Demucs checkpoints are not supported without diffq") from exc
        raise


def _drop_unsupported_kwargs(klass, kwargs):
    import inspect

    signature = inspect.signature(klass)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _build_model_from_package(package, model_path=None):
    if isinstance(package, tuple) and len(package) >= 4:
        klass, args, kwargs, state = package[:4]
    elif isinstance(package, dict) and {"klass", "args", "kwargs", "state"} <= set(package):
        klass = package["klass"]
        args = package["args"]
        kwargs = package["kwargs"]
        state = package["state"]
    elif isinstance(package, dict):
        if model_path is None:
            raise ValueError("state_dict-only legacy Demucs checkpoint requires model_path for architecture inference")
        klass, args, kwargs = _infer_state_dict_architecture(package, model_path)
        state = package
    else:
        raise ValueError(f"Unsupported legacy Demucs checkpoint format: {type(package).__name__}")

    model_cls = _resolve_klass(klass)
    kwargs = _drop_unsupported_kwargs(model_cls, dict(kwargs))
    model = model_cls(*args, **kwargs)
    if isinstance(state, dict) and state.get("__quantized"):
        raise ValueError("DiffQ quantized legacy Demucs checkpoints are not supported without diffq")
    model.load_state_dict(state)
    _ensure_legacy_metadata(model)
    return model


def _ensure_legacy_metadata(model):
    if not hasattr(model, "segment_length"):
        if hasattr(model, "segment") and hasattr(model, "samplerate"):
            model.segment_length = int(float(model.segment) * model.samplerate)
        else:
            model.segment_length = 44100 * 10
    if not hasattr(model, "samplerate"):
        model.samplerate = 44100
    if not hasattr(model, "audio_channels"):
        model.audio_channels = 2
    return model


def _infer_state_dict_architecture(state, model_path):
    name = Path(model_path).stem
    if "encoder.conv1d_U.weight" in state and "decoder.basis_signals.weight" in state:
        encoder = state["encoder.conv1d_U.weight"]
        mask_conv = state["separator.network.3.weight"]
        repeats = {
            int(key.split(".")[3]) for key in state if key.startswith("separator.network.2.") and len(key.split(".")) > 4
        }
        blocks = {
            int(key.split(".")[4]) for key in state if key.startswith("separator.network.2.0.") and len(key.split(".")) > 5
        }
        klass = _stub_class("demucs.tasnet", "ConvTasNet")
        return (
            klass,
            (),
            {
                "sources": int(mask_conv.shape[0] // encoder.shape[0]),
                "N": int(encoder.shape[0]),
                "L": int(encoder.shape[2]),
                "B": int(state["separator.network.1.weight"].shape[0]),
                "H": int(state["separator.network.2.0.0.net.0.weight"].shape[0]),
                "P": int(state["separator.network.2.0.0.net.3.net.0.weight"].shape[-1]),
                "X": max(blocks) + 1 if blocks else 8,
                "R": max(repeats) + 1 if repeats else 4,
                "audio_channels": int(encoder.shape[1]),
            },
        )
    if name.startswith("demucs_unittest"):
        klass = _stub_class("demucs.model", "Demucs")
        depth = sum(1 for key in state if key.startswith("encoder.") and key.endswith(".0.weight"))
        return klass, (), {"sources": 4, "audio_channels": 2, "channels": 4, "depth": depth, "lstm_layers": 2}
    if "lstm.lstm.weight_ih_l0" in state and "encoder.0.0.weight" in state:
        first = state["encoder.0.0.weight"]
        last_decoder_bias = (
            f"decoder.{sum(1 for key in state if key.startswith('encoder.') and key.endswith('.0.weight')) - 1}.2.bias"
        )
        sources = state[last_decoder_bias].numel() // first.shape[1] if last_decoder_bias in state else 4
        channels = first.shape[0]
        depth = sum(1 for key in state if key.startswith("encoder.") and key.endswith(".0.weight"))
        context = int(state["decoder.0.0.weight"].shape[-1]) if "decoder.0.0.weight" in state else 3
        deepest_encoder = f"encoder.{depth - 1}.0.weight"
        resample = (
            bool(state["decoder.0.2.weight"].shape[1] == state[deepest_encoder].shape[0])
            if "decoder.0.2.weight" in state and deepest_encoder in state
            else False
        )
        klass = _stub_class("demucs.model", "Demucs")
        return (
            klass,
            (),
            {
                "sources": int(sources),
                "audio_channels": int(first.shape[1]),
                "channels": int(channels),
                "depth": int(depth),
                "lstm_layers": 2,
                "context": context,
                "resample": resample,
            },
        )
    raise ValueError(f"Cannot infer legacy Demucs architecture from state_dict-only checkpoint: {model_path}")


def load_legacy_demucs_model(model_path, config_path=None):
    path = Path(model_path)
    bag_path = path if path.suffix.lower() in {".yaml", ".yml"} else Path(config_path) if config_path else None
    if bag_path and bag_path.exists():
        bag = yaml.safe_load(bag_path.read_text(encoding="utf-8"))
        if isinstance(bag, dict) and "models" in bag:
            models = []
            for name in bag["models"]:
                candidates = sorted(bag_path.parent.glob(f"{name}*.th"))
                if not candidates:
                    raise FileNotFoundError(f"Cannot find legacy Demucs bag member {name!r} next to {bag_path}")
                models.append(load_legacy_demucs_model(candidates[0])[0])
            return LegacyBagOfModels(models, bag.get("weights"), bag.get("segment")), _legacy_config_from_model(models[0])
        if path == bag_path:
            raise ValueError(f"Legacy Demucs YAML must contain a 'models' list: {bag_path}")

    model = _build_model_from_package(_load_raw_checkpoint(path), path)
    return model, _legacy_config_from_model(model)


def _legacy_config_from_model(model):
    _ensure_legacy_metadata(model)
    sources = list(model.sources)
    return {
        "model": {
            "stereo": model.audio_channels == 2,
            "legacy_demucs": True,
        },
        "training": {
            "instruments": sources,
            "target_instrument": None,
            "samplerate": int(model.samplerate),
            "segment": float(model.segment_length) / float(model.samplerate),
            "channels": int(model.audio_channels),
            "use_amp": True,
        },
        "audio": {
            "sample_rate": int(model.samplerate),
            "chunk_size": int(model.segment_length),
        },
        "inference": {
            "batch_size": 1,
            "overlap_size": int(model.segment_length * 0.25),
            "normalize": False,
            "shifts": 0,
            "split": True,
        },
    }
