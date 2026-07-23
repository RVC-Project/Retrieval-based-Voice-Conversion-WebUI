import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from .separation import SeparationNet
import math


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class ConvolutionModule(nn.Module):
    def __init__(self, channels, depth=2, compress=4, kernel=3):
        super().__init__()
        assert kernel % 2 == 1
        hidden_size = int(channels / compress)
        norm = lambda d: nn.GroupNorm(1, d)
        padding = kernel // 2
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    norm(channels),
                    nn.Conv1d(channels, hidden_size * 2, kernel, padding=padding),
                    nn.GLU(1),
                    nn.Conv1d(hidden_size, hidden_size, kernel, padding=padding, groups=hidden_size),
                    norm(hidden_size),
                    Swish(),
                    nn.Conv1d(hidden_size, channels, 1),
                )
                for _ in range(abs(depth))
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class FusionLayer(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size, stride=stride, padding=padding)

    def forward(self, x, skip=None):
        if skip is not None:
            x += skip
        return F.glu(self.conv(x.repeat(1, 2, 1, 1)), dim=1)


class SDlayer(nn.Module):
    def __init__(self, channels_in, channels_out, band_configs):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(channels_in, channels_out, (config["kernel"], 1), (config["stride"], 1), (0, 0))
                for config in band_configs.values()
            ]
        )
        self.strides = [config["stride"] for config in band_configs.values()]
        self.kernels = [config["kernel"] for config in band_configs.values()]
        self.SR_low = band_configs["low"]["SR"]
        self.SR_mid = band_configs["mid"]["SR"]

    def forward(self, x):
        Fr = x.shape[2]
        low, mid = math.ceil(Fr * self.SR_low), math.ceil(Fr * (self.SR_low + self.SR_mid))
        splits = [(0, low), (low, mid), (mid, Fr)]

        outputs, original_lengths = [], []
        for conv, stride, kernel, (start, end) in zip(self.convs, self.strides, self.kernels, splits):
            extracted = x[:, :, start:end, :]
            original_lengths.append(end - start)
            total_padding = kernel - stride if stride == 1 else (stride - extracted.shape[2] % stride) % stride
            pad_left = total_padding // 2
            outputs.append(conv(F.pad(extracted, (0, 0, pad_left, total_padding - pad_left))))

        return outputs, original_lengths


class SUlayer(nn.Module):
    def __init__(self, channels_in, channels_out, band_configs):
        super().__init__()
        self.convtrs = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels_in, channels_out, [config["kernel"], 1], [config["stride"], 1])
                for config in band_configs.values()
            ]
        )

    def forward(self, x, lengths, origin_lengths):
        def upsample(idx, convtr, start, end):
            out = convtr(x[:, :, start:end, :])
            dist = abs(origin_lengths[idx] - out.shape[2]) // 2
            return out[:, :, dist : dist + origin_lengths[idx], :]

        return torch.cat(
            [
                upsample(idx, convtr, start, end)
                for idx, (convtr, (start, end)) in enumerate(
                    zip(self.convtrs, [(0, lengths[0]), (lengths[0], lengths[0] + lengths[1]), (lengths[0] + lengths[1], None)])
                )
            ],
            dim=2,
        )


class SDblock(nn.Module):
    def __init__(self, channels_in, channels_out, band_configs={}, conv_config={}, depths=[3, 2, 1], kernel_size=3):
        super().__init__()
        self.SDlayer = SDlayer(channels_in, channels_out, band_configs)
        self.conv_modules = nn.ModuleList([ConvolutionModule(channels_out, depth, **conv_config) for depth in depths])
        self.globalconv = nn.Conv2d(channels_out, channels_out, kernel_size, 1, (kernel_size - 1) // 2)

    def forward(self, x):
        bands, original_lengths = self.SDlayer(x)
        bands = [
            F.gelu(
                conv(band.permute(0, 2, 1, 3).reshape(-1, band.shape[1], band.shape[3]))
                .view(band.shape[0], band.shape[2], band.shape[1], band.shape[3])
                .permute(0, 2, 1, 3)
            )
            for conv, band in zip(self.conv_modules, bands)
        ]
        lengths = [band.size(-2) for band in bands]
        full_band = torch.cat(bands, dim=2)
        return self.globalconv(full_band), full_band, lengths, original_lengths


class SCNet(nn.Module):
    mps_model_backend = "torch"
    mps_model_compute_dtype = torch.float16

    def __init__(
        self,
        sources=["drums", "bass", "other", "vocals"],
        audio_channels=2,
        dims=[4, 32, 64, 128],
        nfft=4096,
        hop_size=1024,
        win_size=4096,
        normalized=True,
        band_SR=[0.175, 0.392, 0.433],
        band_stride=[1, 4, 16],
        band_kernel=[3, 4, 16],
        conv_depths=[3, 2, 1],
        compress=4,
        conv_kernel=3,
        num_dplayer=6,
        expand=1,
    ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.dims = dims
        band_keys = ["low", "mid", "high"]
        self.band_configs = {
            key: {"SR": sr, "stride": stride, "kernel": kernel}
            for key, sr, stride, kernel in zip(band_keys, band_SR, band_stride, band_kernel)
        }
        self.hop_length = hop_size
        self.conv_config = {"compress": compress, "kernel": conv_kernel}
        self.stft_config = {
            "n_fft": nfft,
            "hop_length": hop_size,
            "win_length": win_size,
            "center": True,
            "normalized": normalized,
        }

        self.encoder = nn.ModuleList(
            [
                SDblock(
                    channels_in=dims[index],
                    channels_out=dims[index + 1],
                    band_configs=self.band_configs,
                    conv_config=self.conv_config,
                    depths=conv_depths,
                )
                for index in range(len(dims) - 1)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    FusionLayer(channels=dims[index + 1]),
                    SUlayer(
                        channels_in=dims[index + 1],
                        channels_out=dims[index] if index != 0 else dims[index] * len(sources),
                        band_configs=self.band_configs,
                    ),
                )
                for index in reversed(range(len(dims) - 1))
            ]
        )

        self.separation_net = SeparationNet(channels=dims[-1], expand=expand, num_layers=num_dplayer)

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

    def _use_mlx_full_forward(self, x):
        return not self.training and self.mps_model_backend == "mlx_full" and x.device.type == "mps"

    def mlx_forward_mx(self, raw_audio):
        from ..scnet_mlx import mlx_forward_scnet_mx

        return mlx_forward_scnet_mx(self, raw_audio, self.mps_model_compute_dtype)

    def forward(self, x):
        if self._use_mlx_full_forward(x):
            try:
                from ..scnet_mlx import mlx_forward_scnet

                return mlx_forward_scnet(self, x, self.mps_model_compute_dtype)
            except Exception as exc:
                self._pymss_mlx_full_backend_error = repr(exc)
                self.mps_model_backend = "torch"
        B = x.shape[0]
        padding = self.hop_length - x.shape[-1] % self.hop_length
        if (x.shape[-1] + padding) // self.hop_length % 2 == 0:
            padding += self.hop_length
        x = F.pad(x, (0, padding))

        L = x.shape[-1]
        x = torch.view_as_real(torch.stft(x.reshape(-1, L), **self.stft_config, return_complex=True))
        x = x.permute(0, 3, 1, 2).reshape(
            x.shape[0] // self.audio_channels, x.shape[3] * self.audio_channels, x.shape[1], x.shape[2]
        )

        B, C, Fr, T = x.shape

        saved = deque()
        for sd_layer in self.encoder:
            x, skip, lengths, original_lengths = sd_layer(x)
            saved.append((skip, lengths, original_lengths))

        x = self.separation_net(x)

        for fusion_layer, su_layer in self.decoder:
            skip, lengths, original_lengths = saved.pop()
            x = su_layer(fusion_layer(x, skip), lengths, original_lengths)

        n = self.dims[0]
        x = torch.istft(
            torch.view_as_complex(x.view(B, n, -1, Fr, T).reshape(-1, 2, Fr, T).permute(0, 2, 3, 1).contiguous()),
            **self.stft_config,
        )
        return x.reshape(B, len(self.sources), self.audio_channels, -1)[:, :, :, :-padding]
