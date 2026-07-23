__all__ = ("MultiMaskMultiSourceBandSplitRNNSimple",)


def __getattr__(name):
    if name == "MultiMaskMultiSourceBandSplitRNNSimple":
        from .bsrnn.wrapper import MultiMaskMultiSourceBandSplitRNNSimple

        return MultiMaskMultiSourceBandSplitRNNSimple
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
