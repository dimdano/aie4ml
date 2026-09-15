from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ...aie_types import FLOAT_FORMATS, AIEDataType, FloatIntent, QuantIntent, RoundingMode

ACC_TAG_WIDTHS = {
    'acc32': 32,
    'acc48': 48,
    'acc64': 64,
    'acc80': 80,
}

ROUNDING_TOKEN_MAP: Dict[RoundingMode, str] = {
    RoundingMode.TRN: 'floor',
    RoundingMode.RND_MIN_INF: 'floor',
    RoundingMode.RND_INF: 'ceil',
    RoundingMode.RND: 'symmetric_inf',
    RoundingMode.TRN_ZERO: 'symmetric_zero',
    RoundingMode.RND_ZERO: 'symmetric_zero',
    RoundingMode.RND_CONV: 'conv_even',
}


def to_quant_intent(precision: Any) -> QuantIntent:
    if isinstance(precision, QuantIntent):
        return precision
    if isinstance(precision, AIEDataType):
        return QuantIntent(
            width=int(precision.width),
            frac=int(precision.frac),
            signed=bool(precision.signed),
            rounding=precision.rounding,
            saturation=precision.saturation,
        )
    raise TypeError(f'Unsupported precision representation {type(precision)}')


def resolve_storage_width(width: int, *, allowed: Tuple[int, ...], namespace: str, layer_name: str) -> int:
    width = int(width)
    if width <= 0:
        raise ValueError(f'{layer_name}: invalid {namespace} width {width}.')
    for candidate in allowed:
        if width <= candidate:
            return candidate
    raise ValueError(f'{layer_name}: {namespace} width {width} exceeds supported widths {allowed}.')


def resolve_storage_dtype(
    intent: QuantIntent,
    *,
    allowed: Tuple[int, ...],
    namespace: str,
    layer_name: str,
) -> AIEDataType:
    storage_width = resolve_storage_width(intent.width, allowed=allowed, namespace=namespace, layer_name=layer_name)
    return AIEDataType(
        format=f'{"int" if intent.signed else "uint"}{storage_width}',
        frac=int(intent.frac),
        rounding=intent.rounding,
        saturation=intent.saturation,
    )


def resolve_exact_storage_dtype(precision: Any, *, namespace: str, layer_name: str) -> AIEDataType:
    if isinstance(precision, FloatIntent):
        return AIEDataType(
            format=precision.format.value,
        )
    return resolve_storage_dtype(
        to_quant_intent(precision), allowed=(4, 8, 16, 32), namespace=namespace, layer_name=layer_name
    )


def infer_accumulator_tag(
    device: Any,
    lhs_dtype: Optional[AIEDataType],
    rhs_dtype: Optional[AIEDataType],
    acc_precision: Optional[AIEDataType],
) -> Optional[str]:
    if acc_precision is not None:
        if acc_precision.format == 'accfloat':
            return 'accfloat'
        for tag, bits in ACC_TAG_WIDTHS.items():
            if bits == int(acc_precision.width):
                return tag
        raise ValueError(
            f'Unsupported accumulator precision width {acc_precision.width}; expected one of 32, 48, 64 or 80 bits.'
        )

    if lhs_dtype is None or rhs_dtype is None:
        return None

    if lhs_dtype.format in FLOAT_FORMATS or rhs_dtype.format in FLOAT_FORMATS:
        if lhs_dtype.format not in FLOAT_FORMATS or rhs_dtype.format not in FLOAT_FORMATS:
            raise ValueError(
                f'No accumulator tag registered for mixed float/integer precisions '
                f'({lhs_dtype.format!r}, {rhs_dtype.format!r}).'
            )
        return 'accfloat'

    lhs_w = int(getattr(lhs_dtype, 'width', 0) or 0)
    rhs_w = int(getattr(rhs_dtype, 'width', 0) or 0)
    norm_gen = (getattr(device, 'generation', '') or '').upper()
    is_ml = norm_gen.startswith('AIE-ML') or 'XDNA' in norm_gen

    if not is_ml:
        if lhs_w <= 8 and rhs_w <= 8:
            return 'acc48'
        if lhs_w <= 16 and rhs_w <= 16:
            return 'acc48'
        if lhs_w <= 32 and rhs_w <= 32:
            return 'acc80'
        raise ValueError(
            f'No accumulator tag registered for AIE generation "{device.generation}" '
            f'with lhs {lhs_w}-bit and rhs {rhs_w}-bit precisions.'
        )

    if max(lhs_w, rhs_w) <= 8:
        return 'acc32'
    if {lhs_w, rhs_w} in ({8, 16}, {16, 8}):
        return 'acc32'
    if max(lhs_w, rhs_w) <= 16:
        return 'acc64'
    if max(lhs_w, rhs_w) <= 32:
        return 'acc64'
    raise ValueError(
        f'No accumulator tag registered for AIE generation "{device.generation}" '
        f'with lhs {lhs_w}-bit and rhs {rhs_w}-bit precisions.'
    )


def aie_rounding_token(source) -> str:
    mode = getattr(source, 'rounding_mode', None) or getattr(source, 'rounding', None) or RoundingMode.TRN
    token = ROUNDING_TOKEN_MAP.get(mode)
    if token is None:
        raise ValueError(f'Unsupported rounding mode {mode} for AIE kernel.')
    return token


def resolve_accumulator_output_shift(
    lhs_precision: Any,
    output_precision: Any,
    rhs_precision: Any = None,
) -> int:
    """Right-shift from accumulator fixed-point to output fixed-point.

    For matmul: acc_frac = lhs_frac + rhs_frac; for elementwise add: rhs_precision=None → rhs_frac=0.
    """
    lhs_frac = to_quant_intent(lhs_precision).frac
    out_frac = to_quant_intent(output_precision).frac
    rhs_frac = to_quant_intent(rhs_precision).frac if rhs_precision is not None else 0
    return max(0, int(lhs_frac + rhs_frac - out_frac))


def element_bytes(dtype: Optional[AIEDataType]) -> int:
    if not dtype or not getattr(dtype, 'width', None):
        return 1
    return max(1, (int(dtype.width) + 7) // 8)


def storage_bytes_for_spec(spec: Any) -> int:
    return max(1, int((int(spec.width) + 7) // 8))
