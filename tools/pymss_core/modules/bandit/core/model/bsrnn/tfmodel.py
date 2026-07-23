from ....tfmodel import (
    ResidualRNN,
    TimeFrequencyModellingModule,
    _SeqBandModellingPreset,
)


class SeqBandModellingModule(_SeqBandModellingPreset):
    pass


__all__ = ("ResidualRNN", "SeqBandModellingModule", "TimeFrequencyModellingModule")
