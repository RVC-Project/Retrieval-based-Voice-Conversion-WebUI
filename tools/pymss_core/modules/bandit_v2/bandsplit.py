from ..bandit.bandsplit import (
    SequentialNormFC as NormFC,
    _ConfiguredBandSplitModule,
)


class BandSplitModule(_ConfiguredBandSplitModule):
    norm_fc_cls = NormFC
    complex_order = "freq_reim"
    flatten_input = True


__all__ = ("BandSplitModule", "NormFC")
