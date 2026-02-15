# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Integer quantization pass for AIE Dense layers."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from hls4ml.model.optimizer.optimizer import ModelOptimizerPass
from hls4ml.model.types import FixedPrecisionType, PrecisionType
from hls4ml.model.types import RoundingMode as HLSRoundingMode
from hls4ml.model.types import SaturationMode as HLSSaturationMode

from ..aie_types import QuantIntent, RoundingMode, SaturationMode
from ..ir import get_backend_context
from .utils import lookup_layer

log = logging.getLogger(__name__)


def dtype_for_precision(width: Optional[int], signed: bool) -> np.dtype:
    """Map precision metadata to a concrete NumPy dtype."""
    if width is None:
        return np.int32

    if signed:
        if width <= 8:
            return np.int8
        if width <= 16:
            return np.int16
        if width <= 32:
            return np.int32
        return np.int64

    if width <= 8:
        return np.uint8
    if width <= 16:
        return np.uint16
    if width <= 32:
        return np.uint32
    return np.uint64


def wrap_to_width(values: np.ndarray, width: int, signed: bool) -> np.ndarray:
    """Wrap integer values to the provided bit width."""
    modulus = 1 << width
    wrapped = np.mod(values, modulus)
    if signed:
        sign_bit = 1 << (width - 1)
        wrapped = np.where(wrapped >= sign_bit, wrapped - modulus, wrapped)
    return wrapped


def apply_rounding(values: np.ndarray, mode: RoundingMode) -> np.ndarray:
    """Apply the requested rounding mode on floating-point values."""
    if mode in (RoundingMode.TRN, RoundingMode.RND_MIN_INF):
        return np.floor(values)
    if mode in (RoundingMode.TRN_ZERO, RoundingMode.RND_ZERO):
        return np.trunc(values)
    if mode == RoundingMode.RND_INF:
        return np.ceil(values)
    if mode == RoundingMode.RND_CONV:
        return np.round(values)
    if mode == RoundingMode.RND:
        return np.where(values >= 0, np.floor(values + 0.5), np.ceil(values - 0.5))

    raise ValueError(f'Unsupported rounding mode {mode}')


def handle_overflow(
    values: np.ndarray,
    width: int,
    signed: bool,
    mode: SaturationMode,
) -> np.ndarray:
    """Apply overflow handling using the requested saturation mode."""
    if mode == SaturationMode.WRAP:
        return wrap_to_width(values, width, signed)

    dtype = dtype_for_precision(width, signed)
    info = np.iinfo(dtype)

    if mode == SaturationMode.SAT:
        return np.clip(values, info.min, info.max)

    if mode == SaturationMode.SAT_ZERO:
        clipped = np.clip(values, 0, info.max)
        clipped[values < 0] = 0
        return clipped

    if mode == SaturationMode.SAT_SYM:
        sym_min = -info.max if info.min < -info.max else info.min
        return np.clip(values, sym_min, info.max)

    raise ValueError(f'Unsupported saturation mode {mode}')


def _fractional_bits(precision: PrecisionType) -> int:
    if precision is None:
        raise ValueError('Missing precision')
    if hasattr(precision, 'fractional') and precision.fractional is not None:
        return int(precision.fractional)
    if isinstance(precision, FixedPrecisionType):
        return int(precision.width - precision.integer)
    raise TypeError(f'Unable to derive fractional bits from precision type {type(precision)}')


def _bitwidth(precision: PrecisionType) -> int:
    if precision is None:
        raise ValueError('Missing precision')
    if not hasattr(precision, 'width'):
        raise TypeError(f'Precision type {type(precision)} missing width')
    return int(precision.width)


def _is_signed(precision: PrecisionType) -> bool:
    if precision is None:
        raise ValueError('Missing precision')
    if not hasattr(precision, 'signed'):
        raise TypeError(f'Precision type {type(precision)} missing signed')
    return bool(precision.signed)


def _to_quant_intent(precision: PrecisionType) -> QuantIntent:
    if precision is None:
        raise ValueError('Missing precision')
    if not hasattr(precision, 'rounding_mode') or precision.rounding_mode is None:
        raise TypeError(f'Precision type {type(precision)} missing rounding_mode')
    if not hasattr(precision, 'saturation_mode') or precision.saturation_mode is None:
        raise TypeError(f'Precision type {type(precision)} missing saturation_mode')

    rounding = _map_rounding_mode(precision.rounding_mode)
    saturation = _map_saturation_mode(precision.saturation_mode)
    return QuantIntent(
        width=_bitwidth(precision),
        frac=_fractional_bits(precision),
        signed=_is_signed(precision),
        rounding=rounding,
        saturation=saturation,
    )


def _map_rounding_mode(mode: HLSRoundingMode) -> RoundingMode:
    return RoundingMode[mode.name]


def _map_saturation_mode(mode: HLSSaturationMode) -> SaturationMode:
    return SaturationMode[mode.name]


def _quantize_to_int(
    array: np.ndarray,
    frac_bits: int,
    target_bits: int,
    signed: bool = True,
    rounding_mode=None,
    saturation_mode=None,
) -> np.ndarray:
    if array is None:
        return None
    scale = 1 << frac_bits if frac_bits > 0 else 1
    scaled = np.asarray(array, dtype=np.float64) * scale
    rounded = apply_rounding(scaled, rounding_mode)
    integers = rounded.astype(np.int64)
    processed = handle_overflow(integers, target_bits, signed, saturation_mode)
    dtype = dtype_for_precision(target_bits, signed)
    return processed.astype(dtype, copy=False)


def _has_data(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (tuple, list)) and len(value) == 0:
        return False
    return True


class IntegerQuantizer(ModelOptimizerPass):
    """Convert Dense layer tensors to integer representations expected by the AIE backend."""

    def __init__(self):
        self.name = 'integer_quantizer'

    def transform(self, model):
        ctx = get_backend_context(model)
        changed = False

        for node in ctx.ir.logical:
            source_layer_name = node.metadata.get('source_layer')
            layer = lookup_layer(model, source_layer_name)
            if layer is None:
                log.warning('Unable to locate source layer "%s" for IR node %s.', source_layer_name, node.name)
                continue

            if node.op_type == 'dense':
                if _has_data(node.artifacts.get('quant_weights')):
                    continue
                if self._quantize_dense_node(model, ctx, node, layer):
                    changed = True
                continue

            if node.op_type == 'activation':
                if self._quantize_activation_node(node, layer):
                    changed = True

        return changed

    def _quantize_dense_node(self, model, ctx, node, layer) -> bool:
        weight_var = layer.weights.get('weight')
        if weight_var is None:
            log.warning('Layer %s has no weights to quantize; skipping.', layer.name)
            return False

        input_var = layer.get_input_variable()
        output_var = layer.get_output_variable()

        weight_precision = getattr(weight_var.type, 'precision', None)
        input_precision = getattr(input_var.type, 'precision', None)
        result_precision = getattr(output_var.type, 'precision', None)
        if input_precision is None or weight_precision is None or result_precision is None:
            raise RuntimeError(f'Layer {layer.name}: missing precision metadata; run hls4ml quantization first.')

        input_intent = _to_quant_intent(input_precision)
        weight_intent = _to_quant_intent(weight_precision)
        output_intent = _to_quant_intent(result_precision)
        accum_frac = int(input_intent.frac + weight_intent.frac)

        quant_weights = _quantize_to_int(
            weight_var.data,
            weight_intent.frac,
            weight_intent.width,
            signed=weight_intent.signed,
            rounding_mode=weight_intent.rounding,
            saturation_mode=weight_intent.saturation,
        )

        bias_var = layer.weights.get('bias')
        bias_intent = None
        if bias_var is not None and bias_var.data is not None:
            bias_precision = getattr(bias_var.type, 'precision', None)
            if bias_precision is None:
                raise RuntimeError(f'Layer {layer.name}: bias data present but precision is missing.')
            bias_intent = _to_quant_intent(bias_precision)
            quant_bias = _quantize_to_int(
                bias_var.data,
                accum_frac,  # Bias is added in the accumulator domain in the AIE dense kernels.
                32,
                signed=bias_intent.signed,
                rounding_mode=bias_intent.rounding,
                saturation_mode=bias_intent.saturation,
            )
        else:
            quant_bias = None

        node.artifacts['quant_weights'] = quant_weights
        node.artifacts['quant_bias'] = quant_bias

        quant_metadata = node.metadata.setdefault('quant', {})
        quant_metadata['input_precision'] = input_intent
        quant_metadata['weight_precision'] = weight_intent
        if bias_intent is not None:
            quant_metadata['bias_precision'] = QuantIntent(
                width=32,
                frac=accum_frac,
                signed=bool(bias_intent.signed),
                rounding=bias_intent.rounding,
                saturation=bias_intent.saturation,
            )

        quant_metadata['output_precision'] = output_intent

        return True

    def _quantize_activation_node(self, node, layer) -> bool:
        input_var = layer.get_input_variable()
        output_var = layer.get_output_variable()

        input_precision = getattr(input_var.type, 'precision', None)
        output_precision = getattr(output_var.type, 'precision', None)
        if input_precision is None or output_precision is None:
            raise RuntimeError(f'Layer {layer.name}: missing activation precision metadata.')

        quant_metadata = node.metadata.setdefault('quant', {})
        quant_metadata['input_precision'] = _to_quant_intent(input_precision)
        quant_metadata['output_precision'] = _to_quant_intent(output_precision)
        return True
