from typing import List, Tuple

from ....bandsplit import NormFC, _ConfiguredBandSplitModule


class BandSplitModule(_ConfiguredBandSplitModule):
    norm_fc_cls = NormFC
    complex_order = "reim_freq"
    flatten_input = False

    def __init__(
        self,
        band_specs: List[Tuple[float, float]],
        emb_dim: int,
        in_channel: int,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
    ) -> None:
        super().__init__(
            band_specs=band_specs,
            emb_dim=emb_dim,
            in_channels=in_channel,
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
        )


__all__ = ("BandSplitModule", "NormFC")
