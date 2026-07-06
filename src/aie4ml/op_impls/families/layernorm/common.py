from __future__ import annotations

import numpy as np

from ...utils.precision import storage_bytes_for_spec

__all__ = [
    'GAMMA_FRAC_BITS',
    'BETA_FRAC_BITS',
    'DEFAULT_ISQRT_NR_ITERS',
    'DEFAULT_USE_AIE_INVSQRT',
    'layernorm_vec_size',
    'pack_layernorm_param',
]

# Fixed-point conventions baked into the integer LayerNorm kernel.
# gamma is multiplied by inv_std (Q15) and right-shifted by GAMMA_SHIFT, so
# gamma must be stored at frac=GAMMA_SHIFT for fscale to land in Q15.
# beta is added to the Q15 accumulator before the final right-shift, so beta
# is stored at frac=NORM_SHIFT (Q15).
GAMMA_FRAC_BITS = 7
BETA_FRAC_BITS = 15
DEFAULT_ISQRT_NR_ITERS = 1
DEFAULT_USE_AIE_INVSQRT = False


def layernorm_vec_size(precision, device) -> int:
    """Vector lane count for the fully-integer LayerNorm kernel.

    The kernel computes sum/sum-of-squares with aie::accum<acc32, VEC> over
    int8 inputs, so VEC is the int8 lane count of acc32 (32 on AIE-ML).
    Float variants will override this when added.
    """
    elem_bytes = storage_bytes_for_spec(precision)
    if elem_bytes <= 1:
        return 32
    return int(device.vector_bytes) // max(1, elem_bytes)


def pack_layernorm_param(
    values,
    *,
    name: str,
    full_inner: int,
    frac: int,
    cas_num: int,
    width: int = 16,
    signed: bool = True,
    dtype=np.int16,
) -> np.ndarray:
    """Quantize a 1-D float param (gamma or beta) and replicate across cas_num kernels."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if int(arr.shape[0]) != int(full_inner):
        raise ValueError(
            f'LayerNorm parameter {name!r} length {arr.shape[0]} does not match full_inner={int(full_inner)}.'
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'LayerNorm parameter {name!r} contains non-finite values.')

    scale = float(1 << int(frac)) if int(frac) > 0 else 1.0
    scaled = np.rint(arr * scale).astype(np.int64, copy=False)

    if signed:
        lo = -(1 << (int(width) - 1))
        hi = (1 << (int(width) - 1)) - 1
    else:
        lo = 0
        hi = (1 << int(width)) - 1
    if np.any((scaled < lo) | (scaled > hi)):
        min_value = float(lo) / scale
        max_value = float(hi) / scale
        raise ValueError(
            f'LayerNorm parameter {name!r} cannot be represented as '
            f'{"signed" if signed else "unsigned"} int{int(width)} Q{int(frac)}: '
            f'value range [{float(np.min(arr))}, {float(np.max(arr))}], '
            f'representable range [{min_value}, {max_value}].'
        )

    packed = scaled.astype(dtype, copy=False)

    length = int(arr.shape[0])
    return np.broadcast_to(packed.reshape(1, length), (int(cas_num), length)).copy()
