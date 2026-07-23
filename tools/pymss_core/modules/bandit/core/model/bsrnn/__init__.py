from abc import ABC
from typing import Iterable, Mapping, Union

from torch import nn


class BandsplitCoreBase(nn.Module, ABC):
    band_split: nn.Module
    tf_model: nn.Module
    mask_estim: Union[nn.Module, Mapping[str, nn.Module], Iterable[nn.Module]]

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def mask(x, m):
        return x * m
