from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ....aie_types import FloatFormat
from ....quant_utils import apply_rounding, dtype_for_precision, handle_overflow

TILING_OPTIONS: Dict[str, Dict[Tuple[Any, Any], List[Tuple[int, int, int]]]] = {
    'AIE': {
        (8, 8): [(2, 8, 8), (2, 16, 8), (4, 8, 4), (4, 8, 8), (4, 16, 4), (4, 16, 8), (8, 8, 4)],
        (16, 8): [(4, 4, 4), (4, 4, 8), (4, 8, 4), (8, 4, 4)],
        (8, 16): [(4, 4, 8), (4, 4, 4), (8, 8, 1)],
        (16, 16): [(4, 4, 8), (2, 4, 8), (4, 2, 8), (4, 4, 4), (8, 8, 1)],
        ('float', 'float'): [(2, 4, 4)],
    },
    'AIE-ML': {
        (8, 8): [(4, 8, 8), (2, 8, 8), (2, 16, 8), (4, 8, 4), (4, 16, 4), (4, 16, 8), (8, 8, 4), (8, 8, 8)],
        (16, 8): [(4, 4, 8), (2, 8, 8), (4, 4, 4), (4, 8, 4), (8, 4, 4), (8, 4, 8)],
        (8, 16): [(4, 4, 4), (4, 4, 8)],
        (16, 16): [(4, 4, 4), (2, 4, 8), (4, 2, 8), (4, 4, 8), (8, 1, 8), (8, 2, 8)],
        ('bfloat16', 'bfloat16'): [(4, 8, 4)],
        ('float', 'float'): [(4, 8, 4)],
    },
    'AIE-MLV2': {
        (8, 8): [(8, 8, 8), (4, 8, 8)],
        (16, 8): [(4, 4, 8), (8, 2, 8)],
        (8, 16): [(4, 4, 8), (8, 2, 8)],
        (16, 16): [(8, 2, 8)],
        ('bfloat16', 'bfloat16'): [(4, 8, 8)],
        ('float', 'float'): [(4, 8, 4)],
    },
}


def select_generation_key(generation: str) -> str:
    norm = (generation or '').upper()
    for key in sorted(TILING_OPTIONS.keys(), key=len, reverse=True):
        if key in norm:
            return key
    return 'AIE'


def tiling_key(dtype) -> Any:
    c_type = getattr(dtype, 'c_type', '') or ''
    if c_type in ('bfloat16', 'float', 'float32'):
        return c_type
    return int(dtype.width)


def np_dtype_for_spec(spec) -> np.dtype:
    c_type = getattr(spec, 'c_type', '') or ''
    if c_type == 'bfloat16':
        return np.uint16
    if c_type in ('float', 'float32'):
        return np.float32
    return np.int8 if int(spec.width) <= 8 else np.int16


def np_bias_dtype_for_spec(spec) -> np.dtype:
    c_type = getattr(spec, 'c_type', '') or ''
    if c_type in ('bfloat16', 'float', 'float32'):
        return np.float32
    return np.int16 if int(spec.width) <= 16 else np.int32


def pack_as_float(array: np.ndarray, fmt: FloatFormat) -> np.ndarray:
    """Cast weight/bias data to the float storage format required by mmul kernels."""
    if array is None:
        return None
    if fmt == FloatFormat.BF16:
        f32 = np.asarray(array, dtype=np.float32)
        return (f32.view(np.uint32) >> 16).astype(np.uint16)
    return np.asarray(array, dtype=np.float32)


def quantize_to_int(
    array: np.ndarray,
    frac_bits: int,
    target_bits: int,
    signed: bool = True,
    rounding_mode=None,
    saturation_mode=None,
) -> np.ndarray:
    """Quantize float weight/bias data to fixed-point integers for mmul kernels."""
    if array is None:
        return None
    scale = 1 << frac_bits if frac_bits > 0 else 1
    scaled = np.asarray(array, dtype=np.float64) * scale
    rounded = apply_rounding(scaled, rounding_mode)
    integers = rounded.astype(np.int64)
    processed = handle_overflow(integers, target_bits, signed, saturation_mode)
    dtype = dtype_for_precision(target_bits, signed)
    return processed.astype(dtype, copy=False)


def pack_mmul_rhs_matrix(
    W,
    *,
    K: int,
    N: int,
    K_slice: int,
    N_slice: int,
    tile_k: int,
    tile_n: int,
    cas_length: int,
    cas_num: int,
    order: str = 'C',
    dtype=None,
):
    assert tile_k > 0 and tile_n > 0
    assert K_slice % tile_k == 0
    assert N_slice % tile_n == 0

    W = np.asarray(W)
    if dtype is not None:
        W = W.astype(dtype, copy=False)
    if W.ndim < 2:
        raise ValueError('W must have at least 2 dimensions')
    W_kn = W.reshape((-1, K, N))[-1]

    tiles_per_k = K_slice // tile_k
    tiles_per_n = N_slice // tile_n
    elements_per_tile = tile_k * tile_n
    flat_len = tiles_per_k * tiles_per_n * elements_per_tile

    packed = np.zeros((cas_num, cas_length, flat_len), dtype=W_kn.dtype)
    tile_buf = np.zeros((tile_k, tile_n), dtype=W_kn.dtype)

    for chain in range(cas_num):
        n_base = chain * N_slice
        for cas in range(cas_length):
            flat = packed[chain, cas]
            tile_idx = 0
            for k_tile in range(tiles_per_k):
                gk = cas * K_slice + k_tile * tile_k
                real_k = max(0, min(tile_k, K - gk))
                for n_tile in range(tiles_per_n):
                    tile_buf.fill(0)
                    gn = n_base + n_tile * tile_n
                    real_n = max(0, min(tile_n, N - gn))
                    if real_k > 0 and real_n > 0:
                        tile_buf[:real_k, :real_n] = W_kn[gk : gk + real_k, gn : gn + real_n]
                    start = tile_idx * elements_per_tile
                    flat[start : start + elements_per_tile] = tile_buf.ravel(order=order)
                    tile_idx += 1

    return packed


def pack_vector_by_n_slice(
    v,
    *,
    N: int,
    N_slice: int,
    cas_num: int,
    dtype=None,
):
    v = np.asarray(v)
    if dtype is not None:
        v = v.astype(dtype, copy=False)
    if v.ndim > 1:
        v = v.reshape((-1,))[:N]
    if v.shape[0] != N:
        raise ValueError(f'Vector length mismatch: got {v.shape[0]}, expected {N}')

    packed = np.zeros((cas_num, N_slice), dtype=v.dtype)
    for chain in range(cas_num):
        n_base = chain * N_slice
        real = max(0, min(N_slice, N - n_base))
        if real > 0:
            packed[chain, :real] = v[n_base : n_base + real]
    return packed
