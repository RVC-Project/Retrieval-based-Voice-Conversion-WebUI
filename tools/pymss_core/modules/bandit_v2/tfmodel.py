from ..bandit.tfmodel import (
    ResidualRNN,
    TimeFrequencyModellingModule,
    Transpose,
    _SeqBandModellingPreset,
)


class SeqBandModellingModule(_SeqBandModellingPreset):
    @staticmethod
    def _preset_runtime_options(n_modules, parallel_mode):
        return {
            "sequential_transpose": not parallel_mode,
            "checkpoint_segments": None if parallel_mode else n_modules,
        }


__all__ = ("ResidualRNN", "SeqBandModellingModule", "TimeFrequencyModellingModule", "Transpose")
