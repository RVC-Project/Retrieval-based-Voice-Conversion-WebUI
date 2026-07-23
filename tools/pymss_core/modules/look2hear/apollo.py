import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _cached_inference_tensor(module, name, tensor, input, version):
    if tensor is None:
        return None
    if tensor.device == input.device and tensor.dtype == input.dtype:
        return tensor

    key = (name, input.device, input.dtype, version)
    cache = getattr(module, "_apollo_inference_cache", None)
    if cache is None:
        cache = {}
        module._apollo_inference_cache = cache
    cached = cache.get(name)
    if cached is not None and cached[0] == key:
        return cached[1]

    casted = tensor.detach().to(device=input.device, dtype=input.dtype)
    cache[name] = (key, casted)
    return casted


def _complex_from_ri(ri, dim):
    real, imag = ri.float().unbind(dim=dim)
    return torch.complex(real, imag)


def _complex_div_by_real(spec, denom):
    return torch.complex(spec.real / denom, spec.imag / denom)


def pointwise_conv1d(input, conv):
    conv_params = conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups
    if conv_params != ((1,), (1,), (0,), (1,), 1):
        return conv(input)
    weight = conv.weight[:, :, 0]
    bias = conv.bias
    if input.is_cuda and input.dtype in (torch.float16, torch.bfloat16) and not torch.is_grad_enabled():
        weight = _cached_inference_tensor(conv, "pointwise_weight", weight, input, conv.weight._version)
        if bias is not None:
            bias = _cached_inference_tensor(conv, "pointwise_bias", bias, input, bias._version)
    output = F.linear(input.transpose(1, 2), weight, bias)
    return output.transpose(1, 2)


class RMSNorm(nn.Module):
    def __init__(self, dimension, groups=1):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = 1e-5

    def forward(self, input):
        B, N, T = input.shape
        assert N % self.groups == 0
        if self.groups == 1 and not torch.is_grad_enabled():
            if input.is_cuda and input.dtype in (torch.float16, torch.bfloat16):
                weight = _cached_inference_tensor(self, "rms_weight", self.weight, input, self.weight._version)
                return F.rms_norm(input.transpose(1, 2), (N,), weight, self.eps).transpose(1, 2)
            input_norm = F.rms_norm(input.transpose(1, 2), (N,), None, self.eps).transpose(1, 2)
            return input_norm.type_as(input) * self.weight.reshape(1, -1, 1)

        input_float = input.reshape(B, self.groups, -1, T).float()
        input_norm = input_float * torch.rsqrt(input_float.pow(2).mean(-2, keepdim=True) + self.eps)

        return input_norm.type_as(input).reshape(B, N, T) * self.weight.reshape(1, -1, 1)


class RMVN(nn.Module):
    def __init__(self, dimension, groups=1):
        super(RMVN, self).__init__()

        self.mean = nn.Parameter(torch.zeros(dimension))
        self.std = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = 1e-5

    def forward(self, input):
        B, N = input.shape[:2]
        assert N % self.groups == 0
        input_reshape = input.reshape(B, self.groups, N // self.groups, -1)
        T = input_reshape.shape[-1]

        input_norm = (input_reshape - input_reshape.mean(2).unsqueeze(2)) / (
            input_reshape.var(2).unsqueeze(2) + self.eps
        ).sqrt()
        input_norm = input_norm.reshape(B, N, T) * self.std.reshape(1, -1, 1) + self.mean.reshape(1, -1, 1)

        return input_norm.reshape(input.shape)


class Roformer(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_head=8, theta=10000, window=10000, input_drop=0.0, attention_drop=0.0, causal=True
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size // num_head
        self.num_head = num_head
        self.theta = theta
        self.window = window
        cos_freq, sin_freq = self._calc_rotary_emb()
        self.register_buffer("cos_freq", cos_freq)
        self.register_buffer("sin_freq", sin_freq)
        self.register_buffer("reverse_sign", torch.tensor([-1, 1]), persistent=False)
        self._rotary_freq_cache = {}

        self.attention_drop = attention_drop
        self.causal = causal
        self.eps = 1e-5

        self.input_norm = RMSNorm(self.input_size)
        self.input_drop = nn.Dropout(p=input_drop)
        self.weight = nn.Conv1d(self.input_size, self.hidden_size * self.num_head * 3, 1, bias=False)
        self.output = nn.Conv1d(self.hidden_size * self.num_head, self.input_size, 1, bias=False)

        self.MLP = nn.Sequential(
            RMSNorm(self.input_size), nn.Conv1d(self.input_size, self.input_size * 8, 1, bias=False), nn.SiLU()
        )
        self.MLP_output = nn.Conv1d(self.input_size * 4, self.input_size, 1, bias=False)

    def _calc_rotary_emb(self):
        theta_i = torch.arange(0, self.hidden_size, 2)[: self.hidden_size // 2] / self.hidden_size
        freq = (1.0 / (self.theta**theta_i)).reshape(1, -1)
        pos = torch.arange(self.window).reshape(-1, 1)
        cos_freq = torch.cos(pos * freq).repeat_interleave(2, dim=-1)
        sin_freq = torch.sin(pos * freq).repeat_interleave(2, dim=-1)

        return cos_freq, sin_freq

    def _add_rotary_emb(self, feature, pos):
        N = feature.shape[-1]

        feature_reshape = feature.reshape(-1, N)
        pos = min(pos, self.window - 1)
        cos_freq = self.cos_freq[pos]
        sin_freq = self.sin_freq[pos]
        reverse_sign = self.reverse_sign.to(device=feature.device, dtype=feature.dtype)
        feature_reshape_neg = (feature_reshape.reshape(-1, N // 2, 2).flip(-1) * reverse_sign).reshape(-1, N)
        feature_rope = feature_reshape * cos_freq.unsqueeze(0) + feature_reshape_neg * sin_freq.unsqueeze(0)

        return feature_rope.reshape(feature.shape)

    def _add_rotary_sequence(self, feature):
        T, N = feature.shape[-2:]
        feature_reshape = feature.reshape(-1, T, N)

        if feature.is_cuda and feature.dtype in (torch.float16, torch.bfloat16) and not torch.is_grad_enabled():
            key = (T, feature.device, feature.dtype)
            cached = self._rotary_freq_cache.get(key)
            if cached is None:
                cached = (
                    self.cos_freq[:T].to(device=feature.device, dtype=feature.dtype),
                    self.sin_freq[:T].to(device=feature.device, dtype=feature.dtype),
                )
                self._rotary_freq_cache[key] = cached
            cos_freq, sin_freq = cached
            output = torch.empty_like(feature_reshape)
            feature_even = feature_reshape[..., 0::2]
            feature_odd = feature_reshape[..., 1::2]
            cos_freq = cos_freq[..., 0::2].unsqueeze(0)
            sin_freq = sin_freq[..., 0::2].unsqueeze(0)
            output[..., 0::2] = feature_even * cos_freq - feature_odd * sin_freq
            output[..., 1::2] = feature_odd * cos_freq + feature_even * sin_freq
            return output.reshape(feature.shape)

        cos_freq = self.cos_freq[:T]
        sin_freq = self.sin_freq[:T]
        reverse_sign = self.reverse_sign.to(device=feature.device, dtype=feature.dtype)
        feature_reshape_neg = (feature_reshape.reshape(-1, N // 2, 2).flip(-1) * reverse_sign).reshape(-1, T, N)
        feature_rope = feature_reshape * cos_freq.unsqueeze(0) + feature_reshape_neg * sin_freq.unsqueeze(0)

        return feature_rope.reshape(feature.shape)

    def forward(self, input):
        B, _, T = input.shape

        qkv = pointwise_conv1d(self.input_drop(self.input_norm(input)), self.weight)
        weight = qkv.reshape(B, self.num_head, self.hidden_size * 3, T).mT
        Q, K, V = torch.split(weight, self.hidden_size, dim=-1)

        Q_rot = self._add_rotary_sequence(Q)
        K_rot = self._add_rotary_sequence(K)

        V_attention = V if not torch.is_grad_enabled() else V.contiguous()
        attention_output = F.scaled_dot_product_attention(
            Q_rot.contiguous(),
            K_rot.contiguous(),
            V_attention,
            dropout_p=self.attention_drop,
            is_causal=self.causal,
        )
        attention_output = attention_output.mT.reshape(B, -1, T)
        output = pointwise_conv1d(attention_output, self.output) + input

        hidden = self.MLP[0](output)
        hidden = pointwise_conv1d(hidden, self.MLP[1])
        hidden = self.MLP[2](hidden)
        gate, z = hidden.chunk(2, dim=1)
        output = output + pointwise_conv1d(F.silu(gate) * z, self.MLP_output)

        return output, (K_rot, V)


class ConvActNorm1d(nn.Module):
    def __init__(self, in_channel, hidden_channel, kernel=7, causal=False):
        super(ConvActNorm1d, self).__init__()

        self.in_channel = in_channel
        self.kernel = kernel
        self.causal = causal
        if not causal:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channel, in_channel, kernel, padding=(kernel - 1) // 2, groups=in_channel),
                RMSNorm(in_channel),
                nn.Conv1d(in_channel, hidden_channel, 1),
                nn.SiLU(),
                nn.Conv1d(hidden_channel, in_channel, 1),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channel, in_channel, kernel, padding=kernel - 1, groups=in_channel),
                RMSNorm(in_channel),
                nn.Conv1d(in_channel, hidden_channel, 1),
                nn.SiLU(),
                nn.Conv1d(hidden_channel, in_channel, 1),
            )

    def forward(self, input):
        output = self.conv[0](input)
        output = self.conv[1](output)
        output = pointwise_conv1d(output, self.conv[2])
        output = self.conv[3](output)
        output = pointwise_conv1d(output, self.conv[4])
        if self.causal:
            output = output[..., : -self.kernel + 1]
        return input + output


class ICB(nn.Module):
    def __init__(self, in_channel, kernel=7, causal=False):
        super(ICB, self).__init__()

        self.blocks = nn.Sequential(
            ConvActNorm1d(in_channel, in_channel * 4, kernel, causal=causal),
            ConvActNorm1d(in_channel, in_channel * 4, kernel, causal=causal),
            ConvActNorm1d(in_channel, in_channel * 4, kernel, causal=causal),
        )

    def forward(self, input):
        return self.blocks(input)


class BSNet(nn.Module):
    def __init__(self, feature_dim, kernel=7):
        super(BSNet, self).__init__()

        self.feature_dim = feature_dim

        self.band_net = Roformer(self.feature_dim, self.feature_dim, num_head=8, window=100, causal=False)
        self.seq_net = ICB(self.feature_dim, kernel=kernel)

    def forward(self, input):
        B, nband, N, T = input.shape

        band_output, _ = self.band_net(input.permute(0, 3, 2, 1).reshape(B * T, -1, nband))
        band_output = band_output.reshape(B, T, -1, nband).permute(0, 3, 2, 1)

        return self.seq_net(band_output.reshape(B * nband, -1, T)).reshape(B, nband, -1, T)


class Apollo(nn.Module):
    mps_model_backend = "torch"
    mps_model_compute_dtype = torch.float16

    def __init__(self, sr: int, win: int, feature_dim: int, layer: int):
        super().__init__()

        self.sr = sr
        self.win = int(sr * win // 1000)
        self.stride = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps
        self.register_buffer("window", torch.hann_window(self.win), persistent=False)
        self._packed_cache = {}

        bandwidth = int(self.win / 160)
        self.band_width = [bandwidth] * 79 + [self.enc_dim - bandwidth * 79]
        self.nband = len(self.band_width)

        self.BN = nn.ModuleList(
            [nn.Sequential(RMSNorm(width * 2 + 1), nn.Conv1d(width * 2 + 1, self.feature_dim, 1)) for width in self.band_width]
        )
        self.net = nn.Sequential(*[BSNet(self.feature_dim) for _ in range(layer)])
        self.output = nn.ModuleList(
            [
                nn.Sequential(RMSNorm(self.feature_dim), nn.Conv1d(self.feature_dim, width * 4, 1), nn.GLU(dim=1))
                for width in self.band_width
            ]
        )

    def set_mps_model_backend(self, backend=None, compute_dtype=None):
        backend = (backend or "torch").lower()
        if backend not in ("torch", "mlx_full"):
            raise ValueError("mps_model_backend must be 'torch' or 'mlx_full'")
        self.mps_model_backend = backend
        if compute_dtype is None:
            return
        if isinstance(compute_dtype, str):
            compute_dtype = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }.get(compute_dtype.lower(), compute_dtype)
        if compute_dtype not in (torch.float16, torch.float32):
            raise ValueError("mps_model_compute_dtype must be 'float16' or 'float32'")
        self.mps_model_compute_dtype = compute_dtype

    def _use_mlx_full_forward(self, input):
        return not self.training and self.mps_model_backend == "mlx_full" and input.device.type == "mps"

    def mlx_forward_mx(self, raw_audio):
        from ..apollo_mlx import mlx_forward_apollo_mx

        return mlx_forward_apollo_mx(self, raw_audio, self.mps_model_compute_dtype)

    def _window(self, input):
        return self.window.to(device=input.device)

    def _uniform_band_prefix(self):
        width = self.band_width[0]
        return next((i for i, band_width in enumerate(self.band_width) if band_width != width), self.nband), width

    def _use_packed_band_ops(self):
        return not self.training and not torch.is_grad_enabled() and self._uniform_band_prefix()[0] > 1

    def _cached_packed_modules(self, name, modules, count):
        conv = modules[0][1]
        key = (name, count, conv.weight.device, conv.weight.dtype)
        cached = self._packed_cache.get(name)
        if cached is not None and cached["key"] == key:
            return cached["norm_weight"], cached["conv_weight"], cached["conv_bias"], cached["groups"], cached["eps"]

        modules = list(modules[:count])
        norm_weight, conv_weight = (
            torch.stack([module[0].weight.detach() for module in modules], dim=0),
            torch.cat([module[1].weight.detach() for module in modules], dim=0),
        )
        cached = {
            "key": key,
            "norm_weight": norm_weight,
            "conv_weight": conv_weight,
            "conv_bias": torch.cat([module[1].bias.detach() for module in modules], dim=0)
            if modules[0][1].bias is not None
            else None,
            "groups": modules[0][0].groups,
            "eps": modules[0][0].eps,
        }
        self._packed_cache[name] = cached
        return norm_weight, conv_weight, cached["conv_bias"], cached["groups"], cached["eps"]

    def _cached_packed_bn(self, count):
        return self._cached_packed_modules("bn", self.BN, count)

    def _cached_packed_output(self, count):
        return self._cached_packed_modules("output", self.output, count)

    @staticmethod
    def _packed_rms_norm(input, weight, groups, eps):
        batch, bands, channels, frames = input.shape
        input_float = input.reshape(batch, bands, groups, channels // groups, frames).float()
        input_norm = input_float * torch.rsqrt(input_float.pow(2).mean(3, keepdim=True) + eps)
        return input_norm.to(dtype=input.dtype).reshape(batch, bands, channels, frames) * weight.reshape(1, bands, channels, 1)

    def _packed_bn_prefix(self, input, count):
        batch, bands, channels, frames = input.shape
        norm_weight, conv_weight, conv_bias, groups, eps = self._cached_packed_bn(count)
        return F.conv1d(
            self._packed_rms_norm(input, norm_weight, groups, eps).reshape(batch, bands * channels, frames),
            conv_weight,
            conv_bias,
            groups=bands,
        ).reshape(batch, bands, self.feature_dim, frames)

    def _packed_output_prefix(self, feature, count, width):
        batch, bands, channels, frames = feature.shape
        norm_weight, conv_weight, conv_bias, groups, eps = self._cached_packed_output(count)
        output = F.conv1d(
            self._packed_rms_norm(feature, norm_weight, groups, eps).reshape(batch, bands * channels, frames),
            conv_weight,
            conv_bias,
            groups=bands,
        ).reshape(batch, bands, width * 4, frames)
        left, right = output.chunk(2, dim=2)
        return (left * torch.sigmoid(right)).reshape(batch, bands, 2, width, frames)

    def spec_band_split(self, input):
        B, nch, nsample = input.shape

        spec = torch.stft(
            input.view(B * nch, nsample),
            n_fft=self.win,
            hop_length=self.stride,
            window=self._window(input),
            return_complex=True,
        )

        subband_spec_norm, subband_power, band_idx = [], [], 0
        for width in self.band_width:
            this_spec = spec[:, band_idx : band_idx + width]
            subband_power.append((this_spec.abs().pow(2).sum(1) + self.eps).sqrt().unsqueeze(1))  # B, 1, T
            subband_spec_norm.append(_complex_div_by_real(this_spec, subband_power[-1]))  # B, BW, T
            band_idx += width

        return subband_spec_norm, torch.cat(subband_power, 1)

    def _spec_band_split_packed(self, input):
        B, nch, nsample = input.shape
        spec = torch.stft(
            input.view(B * nch, nsample),
            n_fft=self.win,
            hop_length=self.stride,
            window=self._window(input),
            return_complex=True,
        )

        count, width = self._uniform_band_prefix()
        prefix_bins = count * width
        prefix_spec = spec[:, :prefix_bins].reshape(B * nch, count, width, -1)
        prefix_power = (prefix_spec.abs().pow(2).sum(2) + self.eps).sqrt()
        prefix_norm = _complex_div_by_real(prefix_spec, prefix_power.unsqueeze(2))

        tail_norm, tail_power, band_idx = [], [], prefix_bins
        for width in self.band_width[count:]:
            this_spec = spec[:, band_idx : band_idx + width]
            power = (this_spec.abs().pow(2).sum(1) + self.eps).sqrt().unsqueeze(1)
            tail_power.append(power)
            tail_norm.append(_complex_div_by_real(this_spec, power))
            band_idx += width

        return prefix_norm, prefix_power, tail_norm, tail_power

    def feature_extractor(self, input):
        if self._use_packed_band_ops():
            return self._feature_extractor_packed(input)

        return self._feature_extractor_by_band(input)

    def _feature_extractor_by_band(self, input):
        subband_spec_norm, subband_power = self.spec_band_split(input)
        return torch.stack(
            [
                self.BN[i](
                    torch.cat(
                        [subband_spec_norm[i].real, subband_spec_norm[i].imag, torch.log(subband_power[:, i].unsqueeze(1))], 1
                    )
                )
                for i in range(self.nband)
            ],
            1,
        )

    def _feature_extractor_packed(self, input):
        prefix_norm, prefix_power, tail_norm, tail_power = self._spec_band_split_packed(input)
        count, _ = self._uniform_band_prefix()

        prefix_feature = self._packed_bn_prefix(
            torch.cat([prefix_norm.real, prefix_norm.imag, torch.log(prefix_power).unsqueeze(2)], dim=2), count
        )

        if count == self.nband:
            return prefix_feature

        return torch.cat(
            [
                prefix_feature,
                torch.stack(
                    [
                        self.BN[count + offset](
                            torch.cat([subband_norm.real, subband_norm.imag, torch.log(tail_power[offset])], 1)
                        )
                        for offset, subband_norm in enumerate(tail_norm)
                    ],
                    1,
                ),
            ],
            dim=1,
        )

    def _estimate_spec_by_band(self, feature, batch_channels):
        return torch.cat(
            [
                _complex_from_ri(output(feature[:, i]).view(batch_channels, 2, width, -1), dim=1)
                for i, (output, width) in enumerate(zip(self.output, self.band_width))
            ],
            1,
        )

    def _estimate_spec_packed(self, feature, batch_channels):
        count, width = self._uniform_band_prefix()
        prefix_RI = self._packed_output_prefix(feature[:, :count], count, width)
        prefix_spec = _complex_from_ri(prefix_RI, dim=2).reshape(batch_channels, count * width, -1)

        if count == self.nband:
            return prefix_spec

        return torch.cat(
            [
                prefix_spec,
                *[
                    _complex_from_ri(self.output[i](feature[:, i]).view(batch_channels, 2, self.band_width[i], -1), dim=1)
                    for i in range(count, self.nband)
                ],
            ],
            1,
        )

    def forward(self, input):
        if self._use_mlx_full_forward(input):
            try:
                from ..apollo_mlx import mlx_forward_apollo

                return mlx_forward_apollo(self, input, self.mps_model_compute_dtype)
            except Exception as exc:
                self._pymss_mlx_full_backend_error = repr(exc)
                self.mps_model_backend = "torch"

        B, nch, nsample = input.shape

        feature = self.net(self.feature_extractor(input))
        if self._use_packed_band_ops():
            est_spec = self._estimate_spec_packed(feature, B * nch)
        else:
            est_spec = self._estimate_spec_by_band(feature, B * nch)
        return torch.istft(
            est_spec.to(dtype=torch.complex64),
            n_fft=self.win,
            hop_length=self.stride,
            window=self._window(input),
            length=nsample,
        ).view(B, nch, -1)
