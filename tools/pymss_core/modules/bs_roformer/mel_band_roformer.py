import torch
from torch import nn
from torch.nn import Module

from typing import Callable, Optional

from .._dsp import mel_filterbank
from .common import (
    MaskEstimator,
    RoformerRuntimeMixin,
    forward_roformer_mask_core,
    forward_spectral_roformer,
    ignore_roformer_training_kwargs,
    init_roformer_band_modules,
    init_roformer_layers,
    init_roformer_runtime,
    init_roformer_stft,
    roformer_stft_freq_bins,
    roformer_transformer_kwargs,
)


class MelBandRoformer(RoformerRuntimeMixin, Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        num_bands=60,
        dim_head=64,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        flash_attn=True,
        sample_rate=44100,
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn: Optional[Callable] = None,
        zero_dc=True,
        mask_estimator_depth=1,
        match_input_audio_length=False,
        mlp_expansion_factor=4,
        mlp_hidden_layers=None,
        **kwargs,
    ):
        super().__init__()
        ignore_roformer_training_kwargs(kwargs)
        init_roformer_runtime(self, stereo, num_stems)

        transformer_kwargs = roformer_transformer_kwargs(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
        )

        init_roformer_layers(
            self,
            depth=depth,
            time_transformer_depth=time_transformer_depth,
            freq_transformer_depth=freq_transformer_depth,
            dim_head=dim_head,
            transformer_kwargs=transformer_kwargs,
        )

        self.final_norm = nn.Identity()
        init_roformer_stft(self, stft_n_fft, stft_hop_length, stft_win_length, stft_normalized, stft_window_fn)

        freqs = roformer_stft_freq_bins(self, stft_n_fft)

        mel_filter_bank_numpy = mel_filterbank(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)

        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)
        mel_filter_bank[0][0] = 1.0
        mel_filter_bank[-1, -1] = 1.0

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), "all frequencies need to be covered by all bands for now"

        repeated_freq_indices = torch.arange(freqs).expand(num_bands, -1)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = (freq_indices[:, None] * 2 + torch.arange(2)).flatten()

        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("freqs_per_band", freqs_per_band, persistent=False)

        num_freqs_per_band = freqs_per_band.sum(dim=1)
        num_bands_per_freq = freqs_per_band.sum(dim=0)

        self.register_buffer("num_freqs_per_band", num_freqs_per_band, persistent=False)
        self.register_buffer("num_bands_per_freq", num_bands_per_freq, persistent=False)
        self.register_buffer(
            "num_bands_per_channel_freq",
            num_bands_per_freq.repeat_interleave(self.audio_channels).view(1, 1, -1, 1),
            persistent=False,
        )

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in num_freqs_per_band.tolist())
        init_roformer_band_modules(
            self,
            dim=dim,
            freqs_per_bands_with_complex=freqs_per_bands_with_complex,
            num_stems=num_stems,
            mask_estimator_cls=MaskEstimator,
            mask_estimator_depth=mask_estimator_depth,
            mlp_expansion_factor=mlp_expansion_factor,
            mask_estimator_kwargs={"mlp_hidden_layers": mlp_hidden_layers},
        )

        self.zero_dc = zero_dc
        self.match_input_audio_length = match_input_audio_length

    def _forward_mask_core(self, selected_stft_repr):
        return forward_roformer_mask_core(self, selected_stft_repr)

    def _mask_stft_repr(self, stft_repr, context):
        x = stft_repr[torch.arange(context.batch, device=stft_repr.device)[..., None], self.freq_indices]
        self._warm_group_cache(x)
        masks = self._forward_mask_core(x)

        if context.x_is_dml:
            masks = masks.float().cpu()
            stft_repr = torch.view_as_complex(stft_repr.float().cpu().unsqueeze(1).contiguous())
            num_stems = self._active_source_count()
            masks = torch.view_as_complex(masks.contiguous()).to(dtype=stft_repr.dtype)
            scatter_indices = self.freq_indices.cpu()[None, None, :, None].expand(
                context.batch,
                num_stems,
                -1,
                stft_repr.shape[-1],
            )
            masks_summed = stft_repr.new_zeros(
                context.batch,
                num_stems,
                stft_repr.shape[2],
                stft_repr.shape[-1],
            )
            masks_summed.scatter_add_(2, scatter_indices, masks)
            denominator = self.num_bands_per_channel_freq.float().cpu().clamp(min=1e-8)
            return stft_repr * (masks_summed / denominator)

        stft_repr = torch.view_as_complex(stft_repr.unsqueeze(1))
        num_stems = self._active_source_count()
        if stft_repr.device.type == "mps":
            masks = masks.contiguous().to(dtype=stft_repr.real.dtype)
            scatter_indices = self.freq_indices[None, None, :, None, None].expand(
                context.batch,
                num_stems,
                -1,
                stft_repr.shape[-1],
                2,
            )
            masks_summed = masks.new_zeros(context.batch, num_stems, stft_repr.shape[2], stft_repr.shape[-1], 2)
            masks_summed.scatter_add_(2, scatter_indices, masks)
            masks_summed = torch.view_as_complex(masks_summed.contiguous())
        else:
            masks = torch.view_as_complex(masks.contiguous()).to(dtype=stft_repr.dtype)
            scatter_indices = self.freq_indices[None, None, :, None].expand(
                context.batch,
                num_stems,
                -1,
                stft_repr.shape[-1],
            )
            masks_summed = stft_repr.new_zeros(context.batch, num_stems, stft_repr.shape[2], stft_repr.shape[-1])
            masks_summed.scatter_add_(2, scatter_indices, masks)
        return stft_repr * (masks_summed / self.num_bands_per_channel_freq.clamp(min=1e-8))

    def forward(self, raw_audio):
        if self._use_mlx_full_forward(raw_audio):
            try:
                from .mlx_roformer import mlx_forward_roformer

                return mlx_forward_roformer(self, raw_audio, self.mps_model_compute_dtype)
            except Exception as exc:
                self._pymss_mlx_full_backend_error = repr(exc)
                self.mps_model_backend = "torch"
        return forward_spectral_roformer(
            self,
            raw_audio,
            match_input_audio_length=self.match_input_audio_length,
        )
