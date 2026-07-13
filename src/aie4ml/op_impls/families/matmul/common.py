from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ....aie_types import FLOAT_FORMATS
from ....quant_utils import apply_rounding, dtype_for_precision, handle_overflow
from ...utils import AxisPlan, TensorView, build_staging_descriptor, canonical_buffer_axes

# Keys are canonical format-string pairs (lhs_format, rhs_format).
# Integer formats: 'int8', 'int16' (sign-agnostic — both int8_t and uint8_t map here).
# Float formats: 'bfloat16', 'float32', 'fp8_e4m3'.
MICROTILE_OPTIONS: Dict[str, Dict[Tuple[str, str], List[Tuple[int, int, int]]]] = {
    'AIE': {
        ('int8', 'int8'): [(2, 8, 8), (2, 16, 8), (4, 8, 4), (4, 8, 8), (4, 16, 4), (4, 16, 8), (8, 8, 4)],
        ('int16', 'int8'): [(4, 4, 4), (4, 4, 8), (4, 8, 4), (8, 4, 4)],
        ('int8', 'int16'): [(4, 4, 8), (4, 4, 4), (8, 8, 1)],
        ('int16', 'int16'): [(4, 4, 8), (2, 4, 8), (4, 2, 8), (4, 4, 4), (8, 8, 1)],
        ('float32', 'float32'): [(2, 4, 4)],
    },
    'AIE-ML': {
        ('int8', 'int8'): [(4, 8, 8), (2, 8, 8), (2, 16, 8), (4, 8, 4), (4, 16, 4), (4, 16, 8), (8, 8, 4), (8, 8, 8)],
        ('int16', 'int8'): [(4, 4, 8), (2, 8, 8), (4, 4, 4), (4, 8, 4), (8, 4, 4), (8, 4, 8)],
        ('int8', 'int16'): [(4, 4, 4), (4, 4, 8)],
        ('int16', 'int16'): [(4, 4, 4), (2, 4, 8), (4, 2, 8), (4, 4, 8), (8, 1, 8), (8, 2, 8)],
        ('bfloat16', 'bfloat16'): [(4, 8, 4)],
        ('float32', 'float32'): [(4, 8, 4)],
    },
    'AIE-MLV2': {
        ('int8', 'int8'): [(8, 8, 8), (4, 8, 8)],
        ('int16', 'int8'): [(4, 4, 8), (8, 2, 8)],
        ('int8', 'int16'): [(4, 4, 8), (8, 2, 8)],
        ('int16', 'int16'): [(8, 2, 8)],
        ('bfloat16', 'bfloat16'): [(4, 8, 8)],
        ('float32', 'float32'): [(4, 8, 4)],
        ('fp8_e4m3', 'fp8_e4m3'): [(8, 8, 8)],
    },
}


def select_generation_key(generation: str) -> str:
    norm = (generation or '').upper()
    for key in sorted(MICROTILE_OPTIONS.keys(), key=len, reverse=True):
        if key in norm:
            return key
    return 'AIE'


def describe_family_lhs_staging(view: TensorView, microtiling, port: int, buf_dims=None):
    """Staging descriptor for an LHS (activation input) tensor."""
    microtile_m = int(microtiling.microtile_m)
    microtile_k = int(microtiling.microtile_k)
    in_slice = view.tile_inner
    outer_slice = view.tile_outer
    inner_dim, outer_dim, traversal_dims = canonical_buffer_axes(view)
    return build_staging_descriptor(
        view,
        access='read',
        plans={
            inner_dim: AxisPlan(microtile_k, microtile_k, in_slice // microtile_k, port * in_slice),
            outer_dim: AxisPlan(microtile_m, microtile_m, outer_slice // microtile_m),
        },
        order=traversal_dims,
        io_tiling_overrides={inner_dim: view.tile_raw_inner},
        buf_dims=buf_dims,
        boundary_shape='logical',
    )


def describe_family_output_staging(view: TensorView, microtiling, port: int, buf_dims=None):
    """Staging descriptor for an output tensor."""
    microtile_m = int(microtiling.microtile_m)
    microtile_n = int(microtiling.microtile_n)
    out_slice = view.tile_inner
    outer_slice = view.tile_outer
    inner_dim, outer_dim, traversal_dims = canonical_buffer_axes(view)
    return build_staging_descriptor(
        view,
        access='write',
        plans={
            inner_dim: AxisPlan(microtile_n, microtile_n, out_slice // microtile_n, port * out_slice),
            outer_dim: AxisPlan(microtile_m, microtile_m, outer_slice // microtile_m),
        },
        order=traversal_dims,
        io_tiling_overrides={inner_dim: view.tile_raw_inner},
        buf_dims=buf_dims,
    )


def describe_family_rhs_staging(view: TensorView, microtiling, parallelism, port: int, buf_dims=None):
    """Staging descriptor for an RHS (weight) tensor.

    For matmul rhs: view.tile_outer = K slice per port, view.tile_inner = N slice per port.
    """
    microtile_k = int(microtiling.microtile_k)
    microtile_n = int(microtiling.microtile_n)
    # The rhs view encodes both K and N slices: outer dim = K, inner dim = N.
    k_slice = view.tile_outer
    n_slice = view.tile_inner
    inner_dim, outer_dim, traversal_dims = canonical_buffer_axes(view)

    row = int(port) // int(parallelism.cas_length)
    col = int(port) % int(parallelism.cas_length)

    return build_staging_descriptor(
        view,
        access='read',
        plans={
            inner_dim: AxisPlan(microtile_n, microtile_n, max(1, n_slice // microtile_n), row * n_slice),
            outer_dim: AxisPlan(microtile_k, microtile_k, max(1, k_slice // microtile_k), col * k_slice),
        },
        order=traversal_dims,
        io_tiling_overrides={inner_dim: view.tile_raw_inner, outer_dim: view.tile_raw_outer},
        buf_dims=buf_dims,
        boundary_shape='logical',
        extras={
            'packing': 'mmul_rhs',
            'packing_microtile_k': microtile_k,
            'packing_microtile_n': microtile_n,
        },
    )


def np_dtype_for_spec(spec) -> np.dtype:
    fmt = getattr(spec, 'format', '') or ''
    if fmt == 'bfloat16':
        return np.uint16
    if fmt in ('float32', 'accfloat'):
        return np.float32
    if fmt == 'fp8_e4m3':
        return np.uint8
    if fmt.startswith('uint'):
        return np.uint8 if int(spec.width) <= 8 else np.uint16
    return np.int8 if int(spec.width) <= 8 else np.int16


def np_bias_dtype_for_spec(spec) -> np.dtype:
    fmt = getattr(spec, 'format', '') or ''
    if fmt in FLOAT_FORMATS:
        return np.float32
    return np.int16 if int(spec.width) <= 16 else np.int32


def pack_as_float(array: np.ndarray, fmt) -> np.ndarray:
    """Cast weight/bias data to the float storage format required by mmul kernels."""
    if array is None:
        return None
    from ....aie_types import FloatFormat

    fmt_value = getattr(fmt, 'value', fmt)

    def _float32_to_fp8_scalar(f: np.float32) -> np.uint8:
        h = int(np.float32(f).view(np.uint32))
        h = (h + 0x00080000) & 0xFFFFFFFF
        e = (h & 0x7F800000) >> 23
        m = h & 0x007FFFFF
        sign = (h & 0x80000000) >> 24
        if e > 135:
            result = sign | 0x7F
        elif e > 120:
            result = sign | (((e - 120) << 3) & 0x78) | (m >> 20)
        elif e > 116:
            result = sign | (((0x00780000 + m) >> (140 - e)) + 1) >> 1
        else:
            result = sign
        return np.uint8(result & 0xFF)

    if fmt_value == FloatFormat.BF16.value:
        f32 = np.asarray(array, dtype=np.float32)
        return (f32.view(np.uint32) >> 16).astype(np.uint16)
    if fmt_value == FloatFormat.FP8_E4M3.value:
        vfloat32_to_fp8 = np.vectorize(_float32_to_fp8_scalar, otypes=[np.uint8])
        f32 = np.asarray(array, dtype=np.float32)
        return np.asarray(vfloat32_to_fp8(f32), dtype=np.uint8)
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
    microtile_k: int,
    microtile_n: int,
    cas_length: int,
    cas_num: int,
    order: str = 'C',
    dtype=None,
):
    assert microtile_k > 0 and microtile_n > 0
    assert K_slice % microtile_k == 0
    assert N_slice % microtile_n == 0

    W = np.asarray(W)
    if dtype is not None:
        W = W.astype(dtype, copy=False)
    if W.ndim < 2:
        raise ValueError('W must have at least 2 dimensions')
    W_kn = W.reshape((-1, K, N))[-1]

    tiles_per_k = K_slice // microtile_k
    tiles_per_n = N_slice // microtile_n
    elements_per_tile = microtile_k * microtile_n
    flat_len = tiles_per_k * tiles_per_n * elements_per_tile

    packed = np.zeros((cas_num, cas_length, flat_len), dtype=W_kn.dtype)
    tile_buf = np.zeros((microtile_k, microtile_n), dtype=W_kn.dtype)

    for chain in range(cas_num):
        n_base = chain * N_slice
        for cas in range(cas_length):
            flat = packed[chain, cas]
            tile_idx = 0
            for k_tile in range(tiles_per_k):
                gk = cas * K_slice + k_tile * microtile_k
                real_k = max(0, min(microtile_k, K - gk))
                for n_tile in range(tiles_per_n):
                    tile_buf.fill(0)
                    gn = n_base + n_tile * microtile_n
                    real_n = max(0, min(microtile_n, N - gn))
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
