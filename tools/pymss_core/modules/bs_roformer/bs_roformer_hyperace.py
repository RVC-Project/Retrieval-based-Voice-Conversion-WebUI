from typing import Tuple

import torch

from .bs_roformer import BSRoformer
from .common import MaskEstimator as RoformerMaskEstimator
from .hyperace_segm import SegmModel


class MaskEstimator(RoformerMaskEstimator):
    def __init__(self, dim, dim_inputs: Tuple[int, ...], depth, mlp_expansion_factor=4):
        super().__init__(
            dim=dim,
            dim_inputs=dim_inputs,
            depth=depth,
            mlp_expansion_factor=mlp_expansion_factor,
        )
        self.segm = SegmModel(in_bands=len(dim_inputs), in_dim=dim, out_bins=sum(dim_inputs) // 4)

    def forward(self, x, mode="full"):
        if mode == "no_segm":
            return super().forward(x)
        if mode not in ("full", "segm_only"):
            raise ValueError("HyperACE mask_mode must be one of: full, no_segm, segm_only")

        segm = self.segm(x.permute(0, 3, 1, 2))
        segm = segm.permute(0, 2, 3, 1).reshape(segm.shape[0], segm.shape[2], -1)
        if mode == "segm_only":
            return segm
        return super().forward(x) + segm


class BSRoformerHyperACE(BSRoformer):
    mask_estimator_cls = MaskEstimator
    mask_mode = "no_segm"

    def set_mask_mode(self, mode):
        if mode not in ("full", "no_segm", "segm_only"):
            raise ValueError("HyperACE mask_mode must be one of: full, no_segm, segm_only")
        self.mask_mode = mode

    def _estimate_masks(self, x):
        return torch.stack([fn(x, self.mask_mode) for fn in self._active_mask_estimators()], dim=1)
