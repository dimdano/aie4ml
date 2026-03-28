# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""hls4ml-specific utilities: precision bridge and parameter tensor creation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from hls4ml.model.types import PrecisionType

from ...aie_types import QuantIntent, RoundingMode, SaturationMode
from ...ir import LogicalIR, TensorVar


def _to_quant_intent(precision: PrecisionType) -> QuantIntent:
    return QuantIntent(
        width=int(precision.width),
        frac=int(precision.fractional),
        signed=bool(precision.signed),
        rounding=RoundingMode[precision.rounding_mode.name],
        saturation=SaturationMode[precision.saturation_mode.name],
    )


def _precision_of(var) -> Optional[QuantIntent]:
    precision = getattr(getattr(var, 'type', None), 'precision', None)
    if precision is None:
        return None
    try:
        return _to_quant_intent(precision)
    except Exception:
        return None


def _get_post_activation_precision(layer, model) -> Optional[QuantIntent]:
    """Recover Dense output precision when hls4ml's eliminate_linear_activation removed the
    trailing QActivation before lowering, leaving the output_var at accumulator precision.
    """
    config = model.config
    lnp = getattr(config, 'layer_name_precision', {})
    linear_key = f'{layer.name.lower()}_linear_result'
    if linear_key not in lnp:
        return None

    existing_names = {lr.name.lower() for lr in model.get_layers()}
    keys = list(lnp.keys())
    try:
        start = keys.index(linear_key)
    except ValueError:
        return None

    for key in keys[start + 1 :]:
        if not key.endswith('_result'):
            continue
        val = lnp[key]
        if val == 'auto':
            continue
        candidate_name = key[: -len('_result')]
        if candidate_name in existing_names:
            return None
        try:
            return _to_quant_intent(config.backend.convert_precision_string(val))
        except Exception:
            return None
    return None


def extract_layer_directives(layer, model) -> Dict[str, Any]:
    """Extract normalized compiler directives from the hls4ml config for one layer."""
    directives: Dict[str, Any] = {}
    cfg = model.config.get_layer_config_value

    placement_cfg = cfg(layer, 'placement', {})
    if isinstance(placement_cfg, dict):
        placement: Dict[str, int] = {}
        if 'col' in placement_cfg:
            placement['col'] = int(placement_cfg['col'])
        if 'row' in placement_cfg:
            placement['row'] = int(placement_cfg['row'])
        if placement:
            directives['placement'] = placement

    tiling: Dict[str, int] = {}
    tiling_cfg = cfg(layer, 'tiling', {})
    for key in ('tile_m', 'tile_k', 'tile_n'):
        flat = cfg(layer, key)
        if flat is not None:
            tiling[key] = int(flat)
        elif key in tiling_cfg:
            tiling[key] = int(tiling_cfg[key])
    if tiling:
        directives['tiling'] = tiling

    parallelism: Dict[str, Any] = {}
    parallel_cfg = cfg(layer, 'parallelism', {})
    for key in ('cas_num', 'cas_length'):
        flat = cfg(layer, key)
        if flat is not None:
            parallelism[key] = int(flat)
        elif key in parallel_cfg:
            parallelism[key] = int(parallel_cfg[key])
    if 'parallel_factor' in parallel_cfg:
        parallelism['parallel_factor'] = int(parallel_cfg['parallel_factor'])
    if parallelism:
        directives['parallelism'] = parallelism

    io_route_cfg = cfg(layer, 'io_route', {})
    if io_route_cfg:
        directives['io_route'] = io_route_cfg

    return directives


# TODO: _create_weight_tensors works around hls4ml's ReplaceMultidimensionalDenseWithConv,
# which creates PointwiseConv1D nodes without propagating weight_quantizer or pre-quantized
# data. Fix: either patch multi_dense.py (a) or keep carrying data here (b, current).


def _create_weight_tensors(layer, graph: LogicalIR):
    """Create weight and bias TensorVars for Dense/Conv1D layers and register them in the graph.

    Returns (weight_tv, bias_tv); bias_tv is None if the layer has no bias.
    """
    weight_var = layer.weights.get('weight')
    if weight_var is None:
        raise RuntimeError(f'Layer {layer.name}: Dense node has no weight variable.')
    weight_precision = getattr(weight_var.type, 'precision', None)
    if weight_precision is None:
        raise RuntimeError(f'Layer {layer.name}: weight has no precision metadata.')

    # Dense: hls4ml pre-quantizes weights; Conv1D (from multi_dense): raw floats, need RND_CONV.
    hls_pre_quantized = layer.get_attr('weight_quantizer') is not None
    raw_intent = _to_quant_intent(weight_precision)
    if hls_pre_quantized:
        weight_data, weight_intent = weight_var.data, raw_intent
    else:
        weight_data = getattr(weight_var, 'data_unquantized', weight_var.data)
        weight_intent = QuantIntent(
            width=raw_intent.width,
            frac=raw_intent.frac,
            signed=raw_intent.signed,
            rounding=RoundingMode.RND_CONV,
            saturation=raw_intent.saturation,
        )

    weight_tv = TensorVar(
        name=f'{layer.name}_weight',
        shape=np.asarray(weight_data).shape,
        precision=weight_intent,
        data=weight_data,
    )
    graph.add_tensor(weight_tv)

    bias_var = layer.weights.get('bias')
    if bias_var is None or bias_var.data is None:
        return weight_tv, None

    bias_precision = getattr(bias_var.type, 'precision', None)
    if bias_precision is None:
        raise RuntimeError(f'Layer {layer.name}: bias has no precision metadata.')
    bias_intent = _to_quant_intent(bias_precision)

    if hls_pre_quantized:
        bias_data = bias_var.data
    else:
        # Pre-quantize raw bias with QKeras-compatible rounding so pack re-quantization is idempotent.
        raw_bias = getattr(bias_var, 'data_unquantized', bias_var.data)
        delta = 1.0 / (1 << bias_intent.frac)
        bias_data = np.round(np.asarray(raw_bias, dtype=np.float64) / delta) * delta

    bias_tv = TensorVar(
        name=f'{layer.name}_bias',
        shape=np.asarray(bias_data).shape,
        precision=bias_intent,
        data=bias_data,
    )
    graph.add_tensor(bias_tv)
    return weight_tv, bias_tv
