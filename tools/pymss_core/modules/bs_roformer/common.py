from functools import partial
from typing import NamedTuple

import torch
from torch import nn

from .bands import BandSplit, MaskEstimator
from .transformer import RMSNorm, Transformer


__all__ = (
    "DEFAULT_FREQS_PER_BANDS",
    "MaskEstimator",
    "RMSNorm",
    "RoformerRuntimeMixin",
    "forward_bandsplit_roformer",
    "forward_roformer_mask_core",
    "forward_spectral_roformer",
    "ignore_roformer_training_kwargs",
    "init_roformer_band_modules",
    "init_roformer_layers",
    "init_roformer_runtime",
    "init_roformer_shared_bias",
    "init_roformer_stft",
    "roformer_stft_freq_bins",
    "roformer_transformer_kwargs",
    "roformer_freqs_per_bands_with_complex",
)


DEFAULT_FREQS_PER_BANDS = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)


class SpectralContext(NamedTuple):
    batch: int
    channels: int
    freq_bins: int
    audio_length: int
    stft_window: torch.Tensor
    x_is_mps: bool
    x_is_dml: bool


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)
        self.cache = {}

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return torch.arange(seq_len, device=device, dtype=dtype) + offset

    def forward(self, t, cache_key=None):
        if cache_key in self.cache:
            return self.cache[cache_key]
        t = t() if callable(t) else t
        freqs = (t.to(self.freqs.dtype)[:, None] * self.freqs[None]).repeat_interleave(2, -1)
        if cache_key is not None:
            self.cache[cache_key] = freqs
        return freqs


def default(v, d):
    return v if v is not None else d


def mask_to_complex_shape(mask, complex_dim=2):
    b, n, t, fc = mask.shape
    return mask.reshape(b, n, t, fc // complex_dim, complex_dim).permute(0, 1, 3, 2, 4)


TRAINING_LOSS_KWARGS = frozenset(
    {
        "multi_stft_resolution_loss_weight",
        "multi_stft_resolutions_window_sizes",
        "multi_stft_hop_size",
        "multi_stft_normalized",
        "multi_stft_window_fn",
    }
)
REMOVED_ROFORMER_KWARGS = frozenset(
    {"linear_transformer_depth", "use_torch_checkpoint", "skip_connection", "attention_layout", "dim_freqs_in"}
)


def ignore_roformer_training_kwargs(kwargs):
    unexpected = set(kwargs) - TRAINING_LOSS_KWARGS - REMOVED_ROFORMER_KWARGS
    if unexpected:
        raise TypeError(f"unexpected RoFormer config keys: {sorted(unexpected)}")


def init_roformer_runtime(module, stereo, num_stems):
    module.stereo = stereo
    module.audio_channels = 2 if stereo else 1
    module.num_stems = num_stems


def init_roformer_shared_bias(module, dim, heads, dim_head, use_shared_bias):
    if not use_shared_bias:
        return None, None

    dim_inner = heads * dim_head
    module.linear_62_bias_0 = nn.Parameter(torch.ones(dim_inner * 3))
    module.linear_64_bias_0 = nn.Parameter(torch.ones(dim))
    return module.linear_62_bias_0, module.linear_64_bias_0


def roformer_transformer_kwargs(
    *,
    dim,
    heads,
    dim_head,
    attn_dropout,
    ff_dropout,
    flash_attn,
    norm_output=None,
    shared_qkv_bias=None,
    shared_out_bias=None,
):
    kwargs = dict(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        flash_attn=flash_attn,
    )
    kwargs.update(
        {
            k: v
            for k, v in (
                ("norm_output", norm_output),
                ("shared_qkv_bias", shared_qkv_bias),
                ("shared_out_bias", shared_out_bias),
            )
            if v is not None
        }
    )
    return kwargs


def init_roformer_layers(
    module,
    *,
    depth,
    time_transformer_depth,
    freq_transformer_depth,
    dim_head,
    transformer_kwargs,
):
    time_rotary_embed = RotaryEmbedding(dim=dim_head)
    freq_rotary_embed = RotaryEmbedding(dim=dim_head)
    module.layers = nn.ModuleList(
        [
            nn.ModuleList(
                [
                    Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs),
                    Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs),
                ]
            )
            for _ in range(depth)
        ]
    )


def init_roformer_stft(module, stft_n_fft, stft_hop_length, stft_win_length, stft_normalized, stft_window_fn):
    module.stft_kwargs = dict(
        n_fft=stft_n_fft,
        hop_length=stft_hop_length,
        win_length=stft_win_length,
        normalized=stft_normalized,
    )
    module.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)
    module._stft_window_cache = {}


def roformer_stft_freq_bins(module, window_length):
    # The original training code computed this shape through torch.stft on a
    # random probe tensor during model construction. Preserve that RNG-consuming
    # behavior so scratch initialization remains seed-compatible.
    return torch.stft(
        torch.randn(1, 4096),
        **module.stft_kwargs,
        window=torch.ones(window_length),
        return_complex=True,
    ).shape[1]


def roformer_freqs_per_bands_with_complex(module, freqs_per_bands, freqs):
    assert len(freqs_per_bands) > 1
    assert sum(freqs_per_bands) == freqs, (
        f"the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}"
    )
    return tuple(2 * f * module.audio_channels for f in freqs_per_bands)


def init_roformer_band_modules(
    module,
    *,
    dim,
    freqs_per_bands_with_complex,
    num_stems,
    mask_estimator_cls,
    mask_estimator_depth,
    mlp_expansion_factor,
    mask_estimator_kwargs=None,
):
    module.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)
    module.mask_estimators = nn.ModuleList(
        [
            mask_estimator_cls(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
                **(mask_estimator_kwargs or {}),
            )
            for _ in range(num_stems)
        ]
    )


class RoformerRuntimeMixin:
    mps_model_backend = "torch"
    mps_model_compute_dtype = torch.float16

    def set_mps_model_backend(self, backend=None, compute_dtype=None):
        backend = (backend or "torch").lower()
        if backend not in ("torch", "mlx_full"):
            raise ValueError("mps_model_backend must be 'torch' or 'mlx_full'")
        self.mps_model_backend = backend
        if compute_dtype is not None:
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

    def _use_mlx_full_forward(self, raw_audio):
        return not self.training and self.mps_model_backend == "mlx_full" and raw_audio.device.type == "mps"

    def mlx_forward_mx(self, raw_audio):
        from .mlx_roformer import mlx_forward_roformer_mx

        return mlx_forward_roformer_mx(self, raw_audio, self.mps_model_compute_dtype)

    def stft_window(self, device):
        key = (device.type, device.index, torch.float32)
        window = self._stft_window_cache.get(key)
        if window is None or window.device != device:
            window = self.stft_window_fn(device=device)
            self._stft_window_cache[key] = window
        return window

    def _active_source_indices(self):
        indices = getattr(self, "_pymss_source_indices", None)
        return None if indices is None else tuple(int(index) for index in indices)

    def _active_mask_estimators(self):
        indices = self._active_source_indices()
        if indices is None:
            return tuple(self.mask_estimators)
        return tuple(self.mask_estimators[index] for index in indices)

    def _active_source_count(self):
        indices = self._active_source_indices()
        return len(self.mask_estimators) if indices is None else len(indices)

    def _warm_group_cache(self, tensor):
        key = (tensor.device.type, tensor.device.index, tensor.dtype)
        if getattr(self, "_pymss_group_cache_warm_key", None) == key:
            return

        self.band_split.warm_group_cache(tensor.device, tensor.dtype)
        self._pymss_group_cache_warm_key = key

    def _estimate_masks(self, x):
        estimators = self._active_mask_estimators()
        if self._active_source_indices() is not None:
            packed = MaskEstimator.forward_packed_estimators(estimators, x)
            if packed is not None:
                return packed
            return torch.stack([fn(x) for fn in estimators], dim=1)

        use_packed = getattr(self, "_packed_mask_estimators_available", None)
        if use_packed is not False:
            packed = MaskEstimator.forward_packed_estimators(estimators, x)
            if packed is not None:
                self._packed_mask_estimators_available = True
                return packed
            self._packed_mask_estimators_available = False
        return torch.stack([fn(x) for fn in estimators], dim=1)

    def _mask_stft_repr(self, stft_repr, context):
        self._warm_group_cache(stft_repr)
        mask = self._forward_mask_core(stft_repr)
        if context.x_is_dml:
            stft_repr = torch.view_as_complex(stft_repr.float().cpu().unsqueeze(1).contiguous())
            mask = torch.view_as_complex(mask.float().cpu().contiguous()).to(dtype=stft_repr.dtype)
            return stft_repr * mask
        stft_repr = torch.view_as_complex(stft_repr.unsqueeze(1))
        mask = torch.view_as_complex(mask).type(stft_repr.dtype)
        return stft_repr * mask


def forward_roformer_mask_core(module, stft_repr):
    b, fs, model_t, complex_dim = stft_repr.shape
    x = stft_repr.permute(0, 2, 1, 3).reshape(b, model_t, fs * complex_dim)
    x = module.band_split(x)

    for time_transformer, freq_transformer in module.layers:
        b, t, f, d = x.shape
        x = time_transformer(x.permute(0, 2, 1, 3).reshape(b * f, t, d)).reshape(b, f, t, d).permute(0, 2, 1, 3)
        x = freq_transformer(x.reshape(b * t, f, d)).reshape(b, t, f, d)

    return mask_to_complex_shape(module._estimate_masks(module.final_norm(x)), complex_dim=2)


def stft_roformer(module, raw_audio):
    device = raw_audio.device
    x_is_mps = device.type == "mps"
    x_is_dml = device.type == "privateuseone"

    if raw_audio.ndim == 2:
        raw_audio = raw_audio.unsqueeze(1)

    batch, audio_channels, audio_length = raw_audio.shape
    assert (not module.stereo and audio_channels == 1) or (module.stereo and audio_channels == 2), (
        "stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)"
    )

    stft_audio = raw_audio.reshape(batch * audio_channels, audio_length)
    spectral_device = torch.device("cpu") if x_is_dml else device
    stft_window = module.stft_window(spectral_device)

    if x_is_dml:
        stft_repr = torch.stft(
            stft_audio.float().cpu(),
            **module.stft_kwargs,
            window=stft_window,
            return_complex=True,
        )
    else:
        try:
            stft_repr = torch.stft(stft_audio, **module.stft_kwargs, window=stft_window, return_complex=True)
        except RuntimeError:
            stft_repr = torch.stft(
                stft_audio.cpu() if x_is_mps else stft_audio,
                **module.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=True,
            ).to(device)

    stft_repr = torch.view_as_real(stft_repr).reshape(batch, audio_channels, -1, stft_repr.shape[-1], 2)

    b, s, f, t, c = stft_repr.shape
    stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(b, f * s, t, c)
    if x_is_dml:
        model_dtype = next(module.parameters()).dtype
        stft_repr = stft_repr.to(device=device, dtype=model_dtype)
    return stft_repr, SpectralContext(
        batch=batch,
        channels=audio_channels,
        freq_bins=f,
        audio_length=audio_length,
        stft_window=stft_window,
        x_is_mps=x_is_mps,
        x_is_dml=x_is_dml,
    )


def istft_roformer(module, stft_repr, context, length):
    b, n, _, t = stft_repr.shape
    stft_repr = (
        stft_repr.reshape(b, n, context.freq_bins, context.channels, t)
        .permute(0, 1, 3, 2, 4)
        .reshape(b * n * context.channels, context.freq_bins, t)
    )
    if getattr(module, "zero_dc", False):
        stft_repr = stft_repr.index_fill(1, torch.tensor(0, device=stft_repr.device), 0.0)

    if context.x_is_dml:
        recon_audio = torch.istft(
            stft_repr.cpu(),
            **module.stft_kwargs,
            window=context.stft_window.cpu(),
            return_complex=False,
            length=length,
        )
    else:
        try:
            recon_audio = torch.istft(
                stft_repr, **module.stft_kwargs, window=context.stft_window, return_complex=False, length=length
            )
        except RuntimeError:
            recon_audio = torch.istft(
                stft_repr.cpu() if context.x_is_mps else stft_repr,
                **module.stft_kwargs,
                window=context.stft_window.cpu() if context.x_is_mps else context.stft_window,
                return_complex=False,
                length=length,
            ).to(context.stft_window.device)

    recon_audio = recon_audio.reshape(context.batch, n, context.channels, recon_audio.shape[-1])
    return recon_audio[:, 0] if n == 1 else recon_audio


def forward_spectral_roformer(module, raw_audio, match_input_audio_length=True):
    stft_repr, context = stft_roformer(module, raw_audio)
    return istft_roformer(
        module, module._mask_stft_repr(stft_repr, context), context, context.audio_length if match_input_audio_length else None
    )


def forward_bandsplit_roformer(module, raw_audio):
    return forward_spectral_roformer(module, raw_audio, match_input_audio_length=True)
