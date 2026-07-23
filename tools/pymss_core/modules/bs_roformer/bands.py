import os
from collections import defaultdict
from itertools import accumulate
from typing import Tuple

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from .transformer import RMSNorm


EXPERIMENTAL_TRAIN_GROUPED_BANDS_ENV = "PYMSS_CORE_EXPERIMENTAL_TRAIN_GROUPED_BANDS"
EXPERIMENTAL_DEEP_MASK_GROUPING_ENV = "PYMSS_CORE_EXPERIMENTAL_DEEP_MASK_GROUPING"


def experimental_train_grouped_bands():
    return os.environ.get(EXPERIMENTAL_TRAIN_GROUPED_BANDS_ENV, "").lower() in {"1", "true", "yes", "on"}


def experimental_deep_mask_grouping():
    return os.environ.get(EXPERIMENTAL_DEEP_MASK_GROUPING_ENV, "").lower() in {"1", "true", "yes", "on"}


def should_use_grouped_forward(module):
    return module.use_grouped_forward and (not module.training or experimental_train_grouped_bands())


def default(v, d):
    return v if v is not None else d


def dim_input_offsets(dim_inputs):
    return (0, *accumulate(dim_inputs))


def contiguous_dim_groups(dim_inputs):
    groups = []
    start = 0
    for i in range(1, len(dim_inputs) + 1):
        if i == len(dim_inputs) or dim_inputs[i] != dim_inputs[start]:
            groups.append((start, i, dim_inputs[start]))
            start = i
    return tuple(groups)


def grouped_linear(x, weight, bias):
    group_count, out_features, in_features = weight.shape
    leading_shape = x.shape[:-2]
    x = x.reshape(-1, group_count, in_features).transpose(0, 1)
    weight = weight.transpose(1, 2)
    if bias is None:
        out = torch.bmm(x, weight)
    else:
        if bias.dtype != x.dtype or bias.device != x.device:
            bias = bias.to(device=x.device, dtype=x.dtype)
        bias = bias.unsqueeze(1).expand(-1, x.shape[1], -1)
        out = torch.baddbmm(bias, x, weight)
    return out.transpose(0, 1).reshape(*leading_shape, group_count, out_features)


def inference_tanh(x):
    return torch.tanh(x) if torch.is_grad_enabled() else x.tanh_()


def compatible_glu(x, dim=-1):
    if x.device.type == "privateuseone":
        first, second = x.chunk(2, dim=dim)
        return first * second.sigmoid()
    return F.glu(x, dim=dim)


class CompatibleGLU(nn.GLU):
    def forward(self, input):
        return compatible_glu(input, dim=self.dim)


class BandSplit(Module):
    def __init__(self, dim, dim_inputs: Tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self._dim_offsets = dim_input_offsets(dim_inputs)
        self._dim_groups = contiguous_dim_groups(dim_inputs)
        self._group_cache = {}
        self.use_grouped_forward = True
        self.to_features = ModuleList([nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim)) for dim_in in dim_inputs])

    def _get_group_params(self, start, end, device, dtype):
        key = (start, end, device.type, device.index, dtype)
        use_cache = not self.training
        if use_cache:
            cached = self._group_cache.get(key)
            if cached is not None:
                return cached

        norms = [self.to_features[i][0] for i in range(start, end)]
        linears = [self.to_features[i][1] for i in range(start, end)]
        gamma = torch.stack([norm.gamma.to(device=device, dtype=dtype) for norm in norms], dim=0)
        weight = torch.stack([linear.weight.to(device=device, dtype=dtype) for linear in linears], dim=0)
        cached = (
            gamma,
            weight,
            torch.stack([linear.bias.to(device=device, dtype=dtype) for linear in linears], dim=0)
            if linears[0].bias is not None
            else None,
        )
        if use_cache:
            self._group_cache[key] = cached
        return cached

    def _forward_grouped(self, x):
        def forward_group(start, end, dim_in):
            offset_start = self._dim_offsets[start]
            offset_end = self._dim_offsets[end]
            group_x = x[..., offset_start:offset_end].reshape(*x.shape[:-1], end - start, dim_in)
            gamma, weight, bias = self._get_group_params(start, end, x.device, x.dtype)
            if group_x.device.type == "privateuseone" and group_x.dtype == torch.float16:
                group_x = F.normalize(group_x, dim=-1, eps=torch.finfo(group_x.dtype).tiny)
            else:
                group_x = F.normalize(group_x, dim=-1)
            group_x = group_x * (dim_in**0.5) * gamma
            return grouped_linear(group_x, weight, bias)

        return torch.cat([forward_group(start, end, dim_in) for start, end, dim_in in self._dim_groups], dim=-2)

    def warm_group_cache(self, device, dtype):
        if self.training:
            return
        for start, end, _ in self._dim_groups:
            self._get_group_params(start, end, device, dtype)

    def forward(self, x):
        if should_use_grouped_forward(self):
            return self._forward_grouped(x)

        return torch.stack(
            [to_feature(split_input) for split_input, to_feature in zip(x.split(self.dim_inputs, dim=-1), self.to_features)],
            dim=-2,
        )


def MLP(
    dim_in,
    dim_out,
    dim_hidden=None,
    depth=1,
    activation=nn.Tanh,
    hidden_layers=None,
):
    dim_hidden = default(dim_hidden, dim_in)
    hidden_layers = default(hidden_layers, max(depth - 1, 0))
    dims = (dim_in, *((dim_hidden,) * hidden_layers), dim_out)
    return nn.Sequential(
        *[
            layer
            for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:]))
            for layer in (
                (nn.Linear(layer_dim_in, layer_dim_out),)
                if ind == len(dims) - 2
                else (nn.Linear(layer_dim_in, layer_dim_out), activation())
            )
        ]
    )


class MaskEstimator(Module):
    def __init__(
        self,
        dim,
        dim_inputs: Tuple[int, ...],
        depth,
        mlp_expansion_factor=4,
        mlp_hidden_layers=None,
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self._dim_total = sum(dim_inputs)
        self._dim_offsets = dim_input_offsets(dim_inputs)
        self._dim_groups = contiguous_dim_groups(dim_inputs)
        self._group_cache = {}
        self._layer_group_cache = {}
        self._index_cache = {}
        self._packed_layer_group_cache = {}
        self._band_layers_cache = None
        self._band_signatures_cache = None
        self._layer_group_plan = None
        self._layer_group_plan_ready = False
        self._can_group_mlp_cache = None
        self.use_grouped_forward = True
        # _groupable_layers_cache intentionally removed: id()-keyed caching is
        # unsafe under DataParallel replica recycling (see _groupable_layers).
        dim_hidden = dim * mlp_expansion_factor
        self.to_freqs = ModuleList(
            [
                nn.Sequential(
                    MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth, hidden_layers=mlp_hidden_layers), CompatibleGLU(dim=-1)
                )
                for dim_in in dim_inputs
            ]
        )

    def _groupable_layers(self, mlp_with_glu):
        # NOTE: Do not cache by id(mlp_with_glu). Under torch.nn.DataParallel the
        # module is re-replicated on every forward, so band submodules are short
        # lived and Python recycles their id() values. A shared id-keyed cache
        # then returns another band's parsed layers, corrupting per-band output
        # widths. The per-replica memo in _band_groupable_layers is enough.
        if not isinstance(mlp_with_glu, nn.Sequential) or len(mlp_with_glu) != 2:
            return None
        mlp, glu = mlp_with_glu
        if not isinstance(glu, nn.GLU) or not isinstance(mlp, nn.Sequential):
            return None

        layers = []
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                layers.append(("linear", layer))
            elif isinstance(layer, nn.Tanh):
                layers.append(("tanh", None))
            else:
                return None
        if not layers or layers[-1][0] != "linear":
            return None
        return tuple(layers)

    def _band_groupable_layers(self):
        if self._band_layers_cache is None:
            self._band_layers_cache = tuple(self._groupable_layers(mlp_with_glu) for mlp_with_glu in self.to_freqs)
        return self._band_layers_cache

    @staticmethod
    def _layers_signature(layers):
        return tuple(
            item if kind != "linear" else (kind, item.in_features, item.out_features, item.bias is not None)
            for kind, item in layers
        )

    def _band_layer_signatures(self):
        if self._band_signatures_cache is None:
            layers = self._band_groupable_layers()
            self._band_signatures_cache = (
                None
                if any(layer_group is None for layer_group in layers)
                else tuple(self._layers_signature(layer_group) for layer_group in layers)
            )
        return self._band_signatures_cache

    def _layer_grouping_plan(self):
        allow_deep_grouping = self.training and experimental_deep_mask_grouping()
        if self._layer_group_plan_ready and getattr(self, "_layer_group_plan_allow_deep", False) == allow_deep_grouping:
            return self._layer_group_plan

        band_layers = self._band_groupable_layers()
        if any(layers is None for layers in band_layers):
            self._layer_group_plan_ready = True
            self._layer_group_plan_allow_deep = allow_deep_grouping
            self._layer_group_plan = None
            return None

        layer_count = len(band_layers[0])
        if any(len(layers) != layer_count for layers in band_layers):
            self._layer_group_plan_ready = True
            self._layer_group_plan_allow_deep = allow_deep_grouping
            self._layer_group_plan = None
            return None

        plan = []
        for layer_index in range(layer_count):
            first_kind = band_layers[0][layer_index][0]
            if first_kind == "tanh":
                if any(layers[layer_index][0] != "tanh" for layers in band_layers):
                    self._layer_group_plan_ready = True
                    self._layer_group_plan_allow_deep = allow_deep_grouping
                    self._layer_group_plan = None
                    return None
                plan.append(("tanh", None))
                continue

            if first_kind != "linear":
                self._layer_group_plan_ready = True
                self._layer_group_plan_allow_deep = allow_deep_grouping
                self._layer_group_plan = None
                return None

            groups = defaultdict(list)
            for band_index, layers in enumerate(band_layers):
                kind, layer = layers[layer_index]
                if kind != "linear":
                    self._layer_group_plan_ready = True
                    self._layer_group_plan_allow_deep = allow_deep_grouping
                    self._layer_group_plan = None
                    return None
                signature = (layer.in_features, layer.out_features, layer.bias is not None)
                groups[signature].append(band_index)

            plan.append(("linear", tuple((signature, tuple(indices)) for signature, indices in groups.items())))

        # Extra hidden layers in MBR produce large per-band hidden->hidden
        # batched GEMMs. On CUDA those are slower than the existing addmm
        # loop, so keep this fast path to the common two-linear mask heads.
        if sum(1 for kind, _ in plan if kind == "linear") > 2 and not allow_deep_grouping:
            self._layer_group_plan_ready = True
            self._layer_group_plan_allow_deep = allow_deep_grouping
            self._layer_group_plan = None
            return None

        self._layer_group_plan_ready = True
        self._layer_group_plan_allow_deep = allow_deep_grouping
        self._layer_group_plan = tuple(plan)
        return self._layer_group_plan

    def _can_group_mlp(self):
        if self._can_group_mlp_cache is not None:
            return self._can_group_mlp_cache

        signatures = self._band_layer_signatures()
        self._can_group_mlp_cache = signatures is not None and all(signature == signatures[0] for signature in signatures)
        return self._can_group_mlp_cache

    def _indices_tensor(self, indices, device):
        key = (indices, device.type, device.index)
        cached = self._index_cache.get(key)
        if cached is None or cached.device != device:
            cached = torch.as_tensor(indices, device=device)
            self._index_cache[key] = cached
        return cached

    def _get_group_params(self, start, end, device, dtype):
        key = (start, end, device.type, device.index, dtype)
        use_cache = not self.training
        if use_cache:
            cached = self._group_cache.get(key)
            if cached is not None:
                return cached

        grouped_layers = []
        band_layers = self._band_groupable_layers()
        first_layers = band_layers[start]
        for layer_index, (kind, _) in enumerate(first_layers):
            if kind == "tanh":
                grouped_layers.append(("tanh", None, None))
                continue

            linears = [band_layers[i][layer_index][1] for i in range(start, end)]
            weight = torch.stack([linear.weight.to(device=device, dtype=dtype) for linear in linears], dim=0)
            bias = None
            if linears[0].bias is not None:
                bias = torch.stack([linear.bias.to(device=device, dtype=dtype) for linear in linears], dim=0)
            grouped_layers.append(("linear", weight, bias))

        cached = tuple(grouped_layers)
        if use_cache:
            self._group_cache[key] = cached
        return cached

    def _get_layer_group_params(self, layer_index, signature, indices, device, dtype):
        key = (layer_index, signature, indices, device.type, device.index, dtype)
        use_cache = not self.training
        if use_cache:
            cached = self._layer_group_cache.get(key)
            if cached is not None:
                return cached

        band_layers = self._band_groupable_layers()
        linears = [band_layers[i][layer_index][1] for i in indices]
        weight = torch.stack([linear.weight.to(device=device, dtype=dtype) for linear in linears], dim=0)
        bias = None
        if linears[0].bias is not None:
            bias = torch.stack([linear.bias.to(device=device, dtype=dtype) for linear in linears], dim=0)

        cached = (weight, bias)
        if use_cache:
            self._layer_group_cache[key] = cached
        return cached

    def _get_packed_layer_group_params(self, estimators, layer_index, signature, indices, device, dtype):
        estimator_ids = tuple(id(estimator) for estimator in estimators)
        key = (estimator_ids, layer_index, signature, indices, device.type, device.index, dtype)
        use_cache = not any(estimator.training for estimator in estimators)
        if use_cache:
            cached = self._packed_layer_group_cache.get(key)
            if cached is not None:
                return cached

        linears = [
            estimator._band_groupable_layers()[band_index][layer_index][1] for estimator in estimators for band_index in indices
        ]
        weight = torch.stack([linear.weight.to(device=device, dtype=dtype) for linear in linears], dim=0)
        bias = None
        if linears[0].bias is not None:
            bias = torch.stack([linear.bias.to(device=device, dtype=dtype) for linear in linears], dim=0)

        cached = (weight, bias)
        if use_cache:
            self._packed_layer_group_cache[key] = cached
        return cached

    @staticmethod
    def _packable_estimators_by_band(estimators):
        estimators = tuple(estimators)
        if len(estimators) <= 1:
            return False

        first = estimators[0]
        if not isinstance(first, MaskEstimator):
            return False
        if not should_use_grouped_forward(first):
            return False

        first_signatures = first._band_layer_signatures()
        if first_signatures is None:
            return False

        for estimator in estimators[1:]:
            if type(estimator) is not type(first):
                return False
            if not should_use_grouped_forward(estimator):
                return False
            if estimator.dim_inputs != first.dim_inputs:
                return False

            if estimator._band_layer_signatures() != first_signatures:
                return False
        return True

    def _forward_grouped_mlp(self, x):
        def forward_group(start, end):
            group_x = x[:, :, start:end, :]
            for kind, weight, bias in self._get_group_params(start, end, x.device, x.dtype):
                if kind == "linear":
                    group_x = grouped_linear(group_x, weight, bias)
                else:
                    group_x = inference_tanh(group_x)
            return compatible_glu(group_x, dim=-1).flatten(start_dim=-2)

        return torch.cat([forward_group(start, end) for start, end, _ in self._dim_groups], dim=-1)

    def _forward_layer_grouped_mlp(self, x):
        plan = self._layer_grouping_plan()
        if plan is None:
            return None

        group_x = x
        for layer_index, (kind, groups) in enumerate(plan):
            if kind == "tanh":
                group_x = inference_tanh(group_x)
                continue

            out_dims = {signature[1] for signature, _ in groups}
            if len(out_dims) != 1:
                if layer_index != len(plan) - 1:
                    return None
                outs = [None] * len(self.to_freqs)
                for signature, indices in groups:
                    weight, bias = self._get_layer_group_params(layer_index, signature, indices, x.device, x.dtype)
                    band_index = self._indices_tensor(indices, x.device)
                    selected = group_x.index_select(-2, band_index)
                    out = compatible_glu(grouped_linear(selected, weight, bias), dim=-1)
                    for band_position, band_out in zip(indices, out.unbind(dim=-2)):
                        outs[band_position] = band_out
                return torch.cat(outs, dim=-1)

            if group_x.device.type == "privateuseone":
                packed = []
                packed_order = []
                for signature, indices in groups:
                    weight, bias = self._get_layer_group_params(layer_index, signature, indices, x.device, x.dtype)
                    band_index = self._indices_tensor(indices, x.device)
                    selected = group_x.index_select(-2, band_index)
                    packed.append(grouped_linear(selected, weight, bias))
                    packed_order.extend(indices)
                restore_order = [packed_order.index(index) for index in range(len(packed_order))]
                restore_index = torch.tensor(restore_order, device=x.device, dtype=torch.long)
                group_x = torch.cat(packed, dim=-2).index_select(-2, restore_index)
                continue

            next_x = None
            for signature, indices in groups:
                weight, bias = self._get_layer_group_params(layer_index, signature, indices, x.device, x.dtype)
                band_index = self._indices_tensor(indices, x.device)
                selected = group_x.index_select(-2, band_index)
                out = grouped_linear(selected, weight, bias)
                if next_x is None:
                    next_x = out.new_empty(*group_x.shape[:-1], out.shape[-1])
                next_x.index_copy_(-2, band_index, out)
            group_x = next_x

        return compatible_glu(group_x, dim=-1).flatten(start_dim=-2)

    def _forward_by_band_fast(self, x):
        band_layers = self._band_groupable_layers()

        def forward_band(band_index, band_features):
            group_x = band_features
            layers = band_layers[band_index]
            if layers is None:
                return None

            for kind, layer in layers:
                if kind == "tanh":
                    group_x = inference_tanh(group_x)
                else:
                    group_x = layer(group_x)
            return compatible_glu(group_x, dim=-1)

        outs = [forward_band(band_index, band_features) for band_index, band_features in enumerate(x.unbind(dim=-2))]
        return None if any(out is None for out in outs) else torch.cat(outs, dim=-1)

    @staticmethod
    def _packable_estimators(estimators):
        estimators = tuple(estimators)
        if len(estimators) <= 1:
            return False

        first = estimators[0]
        if not isinstance(first, MaskEstimator):
            return False
        if not should_use_grouped_forward(first):
            return False

        first_plan = first._layer_grouping_plan()
        if first_plan is None:
            return False

        for estimator in estimators[1:]:
            if type(estimator) is not type(first):
                return False
            if not should_use_grouped_forward(estimator):
                return False
            if estimator.dim_inputs != first.dim_inputs:
                return False
            if estimator._layer_grouping_plan() != first_plan:
                return False
        return True

    @staticmethod
    def _select_packed_group(group_x, band_index, stem_count):
        if group_x.ndim == 4:
            return group_x.index_select(-2, band_index).unsqueeze(2).expand(-1, -1, stem_count, -1, -1)
        return group_x.index_select(-2, band_index)

    @staticmethod
    def _forward_packed_estimators_two_layer_stream(estimators, x, plan):
        if len(plan) != 3 or plan[0][0] != "linear" or plan[1][0] != "tanh" or plan[2][0] != "linear":
            return None

        first = estimators[0]
        first_groups = plan[0][1]
        final_groups = plan[2][1]
        if len(first_groups) != 1:
            return None

        first_signature, _ = first_groups[0]
        stem_count = len(estimators)
        result = x.new_empty(x.shape[0], stem_count, x.shape[1], sum(first.dim_inputs))

        for final_signature, indices in final_groups:
            weight, bias = first._get_packed_layer_group_params(estimators, 0, first_signature, indices, x.device, x.dtype)
            band_index = first._indices_tensor(indices, x.device)
            group_x = x.index_select(-2, band_index).unsqueeze(2).expand(-1, -1, stem_count, -1, -1)
            b, t, s, g, d = group_x.shape
            group_x = grouped_linear(group_x.reshape(b, t, s * g, d), weight, bias)
            group_x = inference_tanh(group_x)

            weight, bias = first._get_packed_layer_group_params(estimators, 2, final_signature, indices, x.device, x.dtype)
            group_x = grouped_linear(group_x, weight, bias)
            group_x = compatible_glu(group_x, dim=-1).reshape(b, t, s, g, -1)

            if indices == tuple(range(indices[0], indices[-1] + 1)):
                offset_start = first._dim_offsets[indices[0]]
                offset_end = first._dim_offsets[indices[-1] + 1]
                result[:, :, :, offset_start:offset_end] = group_x.flatten(start_dim=-2).transpose(1, 2)
            else:
                for group_position, band_position in enumerate(indices):
                    offset_start = first._dim_offsets[band_position]
                    offset_end = first._dim_offsets[band_position + 1]
                    result[:, :, :, offset_start:offset_end] = group_x[:, :, :, group_position, :].transpose(1, 2)

        return result

    @staticmethod
    def forward_packed_estimators(estimators, x):
        estimators = tuple(estimators)
        if not MaskEstimator._packable_estimators(estimators):
            return MaskEstimator._forward_packed_estimators_by_band(estimators, x)

        first = estimators[0]
        plan = first._layer_grouping_plan()
        streamed = MaskEstimator._forward_packed_estimators_two_layer_stream(estimators, x, plan)
        if streamed is not None:
            return streamed

        stem_count = len(estimators)
        band_count = len(first.to_freqs)
        group_x = x

        for layer_index, (kind, groups) in enumerate(plan):
            if kind == "tanh":
                group_x = inference_tanh(group_x)
                continue

            out_dims = {signature[1] for signature, _ in groups}
            if len(out_dims) != 1:
                if layer_index != len(plan) - 1:
                    return None

                result = x.new_empty(x.shape[0], x.shape[1], stem_count, sum(first.dim_inputs))
                for signature, indices in groups:
                    weight, bias = first._get_packed_layer_group_params(
                        estimators, layer_index, signature, indices, x.device, x.dtype
                    )
                    band_index = first._indices_tensor(indices, x.device)
                    selected = MaskEstimator._select_packed_group(group_x, band_index, stem_count)
                    b, t, s, g, d = selected.shape
                    out = grouped_linear(selected.reshape(b, t, s * g, d), weight, bias)
                    out = compatible_glu(out, dim=-1).reshape(b, t, s, g, -1)
                    if indices == tuple(range(indices[0], indices[-1] + 1)):
                        offset_start = first._dim_offsets[indices[0]]
                        offset_end = first._dim_offsets[indices[-1] + 1]
                        result[:, :, :, offset_start:offset_end] = out.flatten(start_dim=-2)
                    else:
                        for group_position, band_position in enumerate(indices):
                            offset_start = first._dim_offsets[band_position]
                            offset_end = first._dim_offsets[band_position + 1]
                            result[:, :, :, offset_start:offset_end] = out[:, :, :, group_position, :]

                return result.permute(0, 2, 1, 3)

            out_dim = next(iter(out_dims))
            next_x = None
            for signature, indices in groups:
                weight, bias = first._get_packed_layer_group_params(
                    estimators, layer_index, signature, indices, x.device, x.dtype
                )
                band_index = first._indices_tensor(indices, x.device)
                selected = MaskEstimator._select_packed_group(group_x, band_index, stem_count)
                b, t, s, g, d = selected.shape
                out = grouped_linear(selected.reshape(b, t, s * g, d), weight, bias)
                out = out.reshape(b, t, s, g, out_dim)
                if next_x is None:
                    next_x = out.new_empty(x.shape[0], x.shape[1], stem_count, band_count, out_dim)
                next_x.index_copy_(-2, band_index, out)
            group_x = next_x

        out = compatible_glu(group_x, dim=-1).flatten(start_dim=-2)
        return out.permute(0, 2, 1, 3)

    @staticmethod
    def _forward_packed_estimators_by_band(estimators, x):
        estimators = tuple(estimators)
        if not MaskEstimator._packable_estimators_by_band(estimators):
            return None

        first = estimators[0]
        stem_count = len(estimators)
        result = x.new_empty(x.shape[0], stem_count, x.shape[1], first._dim_total)
        band_layers = first._band_groupable_layers()

        for band_index, layers in enumerate(band_layers):
            group_x = x[:, :, band_index, :].unsqueeze(-2).expand(-1, -1, stem_count, -1)
            for layer_index, (kind, layer) in enumerate(layers):
                if kind == "tanh":
                    group_x = inference_tanh(group_x)
                    continue

                signature = (layer.in_features, layer.out_features, layer.bias is not None)
                weight, bias = first._get_packed_layer_group_params(
                    estimators, layer_index, signature, (band_index,), x.device, x.dtype
                )
                group_x = grouped_linear(group_x, weight, bias)

            offset_start = first._dim_offsets[band_index]
            offset_end = first._dim_offsets[band_index + 1]
            result[:, :, :, offset_start:offset_end] = compatible_glu(group_x, dim=-1).transpose(1, 2)

        return result

    def forward(self, x):
        if should_use_grouped_forward(self):
            if self._can_group_mlp():
                return self._forward_grouped_mlp(x)
            grouped = self._forward_layer_grouped_mlp(x)
            if grouped is not None:
                return grouped
            by_band = self._forward_by_band_fast(x)
            if by_band is not None:
                return by_band

        return torch.cat([mlp(band_features) for band_features, mlp in zip(x.unbind(dim=-2), self.to_freqs)], dim=-1)

    def warm_group_cache(self, device, dtype):
        if self.training:
            return
        if self._can_group_mlp():
            for start, end, _ in self._dim_groups:
                self._get_group_params(start, end, device, dtype)
            return

        plan = self._layer_grouping_plan()
        if plan is None:
            return
        for layer_index, (kind, groups) in enumerate(plan):
            if kind == "tanh":
                continue
            for signature, indices in groups:
                self._get_layer_group_params(layer_index, signature, indices, device, dtype)

    @staticmethod
    def warm_packed_estimators(estimators, device, dtype):
        estimators = tuple(estimators)
        if not estimators:
            return
        if any(estimator.training for estimator in estimators):
            return
        if not MaskEstimator._packable_estimators(estimators):
            MaskEstimator._warm_packed_estimators_by_band(estimators, device, dtype)
            return

        first = estimators[0]
        plan = first._layer_grouping_plan()
        if (
            len(plan) == 3
            and plan[0][0] == "linear"
            and plan[1][0] == "tanh"
            and plan[2][0] == "linear"
            and len(plan[0][1]) == 1
        ):
            first_signature, _ = plan[0][1][0]
            for final_signature, indices in plan[2][1]:
                first._get_packed_layer_group_params(estimators, 0, first_signature, indices, device, dtype)
                first._get_packed_layer_group_params(estimators, 2, final_signature, indices, device, dtype)
            return

        for layer_index, (kind, groups) in enumerate(plan):
            if kind == "tanh":
                continue
            for signature, indices in groups:
                first._get_packed_layer_group_params(estimators, layer_index, signature, indices, device, dtype)

    @staticmethod
    def _warm_packed_estimators_by_band(estimators, device, dtype):
        if not MaskEstimator._packable_estimators_by_band(estimators):
            return
        first = estimators[0]
        for band_index, layers in enumerate(first._band_groupable_layers()):
            for layer_index, (kind, layer) in enumerate(layers):
                if kind != "tanh":
                    first._get_packed_layer_group_params(
                        estimators,
                        layer_index,
                        (layer.in_features, layer.out_features, layer.bias is not None),
                        (band_index,),
                        device,
                        dtype,
                    )
