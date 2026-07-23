from torch.nn import Module

from typing import Callable, Optional, Tuple

from .common import (
    DEFAULT_FREQS_PER_BANDS,
    MaskEstimator,
    RMSNorm,
    RoformerRuntimeMixin,
    forward_bandsplit_roformer,
    forward_roformer_mask_core,
    ignore_roformer_training_kwargs,
    init_roformer_band_modules,
    init_roformer_layers,
    init_roformer_runtime,
    init_roformer_shared_bias,
    init_roformer_stft,
    roformer_freqs_per_bands_with_complex,
    roformer_stft_freq_bins,
    roformer_transformer_kwargs,
)


class BSRoformer(RoformerRuntimeMixin, Module):
    mask_estimator_cls = MaskEstimator

    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        flash_attn=True,
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn: Optional[Callable] = None,
        mask_estimator_depth=2,
        mlp_expansion_factor=4,
        use_shared_bias=False,
        **kwargs,
    ):
        super().__init__()
        ignore_roformer_training_kwargs(kwargs)
        init_roformer_runtime(self, stereo, num_stems)

        shared_qkv_bias, shared_out_bias = init_roformer_shared_bias(
            self,
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            use_shared_bias=use_shared_bias,
        )
        transformer_kwargs = roformer_transformer_kwargs(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            norm_output=False,
            shared_qkv_bias=shared_qkv_bias,
            shared_out_bias=shared_out_bias,
        )

        init_roformer_layers(
            self,
            depth=depth,
            time_transformer_depth=time_transformer_depth,
            freq_transformer_depth=freq_transformer_depth,
            dim_head=dim_head,
            transformer_kwargs=transformer_kwargs,
        )

        self.final_norm = RMSNorm(dim)
        init_roformer_stft(self, stft_n_fft, stft_hop_length, stft_win_length, stft_normalized, stft_window_fn)

        freqs = roformer_stft_freq_bins(self, stft_win_length)
        freqs_per_bands_with_complex = roformer_freqs_per_bands_with_complex(self, freqs_per_bands, freqs)
        init_roformer_band_modules(
            self,
            dim=dim,
            freqs_per_bands_with_complex=freqs_per_bands_with_complex,
            num_stems=num_stems,
            mask_estimator_cls=self.mask_estimator_cls,
            mask_estimator_depth=mask_estimator_depth,
            mlp_expansion_factor=mlp_expansion_factor,
        )

    def _forward_mask_core(self, stft_repr):
        return forward_roformer_mask_core(self, stft_repr)

    def forward(self, raw_audio):
        if self._use_mlx_full_forward(raw_audio):
            try:
                from .mlx_roformer import mlx_forward_roformer

                return mlx_forward_roformer(self, raw_audio, self.mps_model_compute_dtype)
            except Exception as exc:
                self._pymss_mlx_full_backend_error = repr(exc)
                self.mps_model_backend = "torch"
        return forward_bandsplit_roformer(self, raw_audio)
