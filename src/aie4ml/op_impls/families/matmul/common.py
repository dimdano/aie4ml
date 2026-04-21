from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ....aie_types import FloatFormat
from ....quant_utils import apply_rounding, dtype_for_precision, handle_overflow
from ...utils import TensorView, canonical_buffer_axes, make_staging_descriptor, ordered_view_shape
from ...utils.precision import storage_bytes_for_spec

MICROTILE_OPTIONS: Dict[str, Dict[Tuple[Any, Any], List[Tuple[int, int, int]]]] = {
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
    for key in sorted(MICROTILE_OPTIONS.keys(), key=len, reverse=True):
        if key in norm:
            return key
    return 'AIE'


def microtile_key(dtype) -> Any:
    c_type = getattr(dtype, 'c_type', '') or ''
    if c_type in ('bfloat16', 'float', 'float32'):
        return c_type
    return int(dtype.width)


def describe_family_lhs_staging(view: TensorView, microtiling, port: int, buf_dims=None):
    """Staging descriptor for an LHS (activation input) tensor."""
    microtile_m = int(microtiling.microtile_m)
    microtile_k = int(microtiling.microtile_k)
    in_slice = view.tile_inner
    raw_in = view.tile_raw_inner
    buffer_dimension = ordered_view_shape(view, 'full') if buf_dims is None else [int(x) for x in buf_dims]
    inner_dim, outer_dim, traversal_dims = canonical_buffer_axes(view)
    io_tiling_dimension = ordered_view_shape(view, 'logical')
    io_tiling_dimension[inner_dim] = raw_in
    tiling_dimension = [1 for _ in buffer_dimension]
    tiling_dimension[inner_dim] = microtile_k
    tiling_dimension[outer_dim] = microtile_m
    tile_traversal = []
    for dim in traversal_dims:
        if dim == inner_dim:
            tile_traversal.append({'dimension': inner_dim, 'stride': microtile_k, 'wrap': in_slice // microtile_k})
        elif dim == outer_dim:
            tile_traversal.append(
                {'dimension': outer_dim, 'stride': microtile_m, 'wrap': buffer_dimension[outer_dim] // microtile_m}
            )
        else:
            tile_traversal.append({'dimension': dim, 'stride': 1, 'wrap': buffer_dimension[dim]})
    offset = [0 for _ in buffer_dimension]
    offset[inner_dim] = port * in_slice
    return make_staging_descriptor(
        access='read',
        view=view,
        tiling_dimension=tiling_dimension,
        offset=offset,
        tile_traversal=tile_traversal,
        inner_dim=inner_dim,
        outer_dim=outer_dim,
        boundary_shape='logical',
        io_boundary_shape='logical',
        io_tiling_dimension=io_tiling_dimension,
    )


def describe_family_output_staging(view: TensorView, microtiling, port: int, buf_dims=None):
    """Staging descriptor for an output tensor."""
    microtile_m = int(microtiling.microtile_m)
    microtile_n = int(microtiling.microtile_n)
    out_slice = view.tile_inner
    raw_out = view.tile_raw_inner
    buffer_dimension = ordered_view_shape(view, 'full') if buf_dims is None else [int(x) for x in buf_dims]
    inner_dim, outer_dim, traversal_dims = canonical_buffer_axes(view)
    io_tiling_dimension = ordered_view_shape(view, 'real')
    io_tiling_dimension[inner_dim] = raw_out
    tiling_dimension = [1 for _ in buffer_dimension]
    tiling_dimension[inner_dim] = microtile_n
    tiling_dimension[outer_dim] = microtile_m
    tile_traversal = []
    for dim in traversal_dims:
        if dim == inner_dim:
            tile_traversal.append({'dimension': inner_dim, 'stride': microtile_n, 'wrap': out_slice // microtile_n})
        elif dim == outer_dim:
            tile_traversal.append(
                {'dimension': outer_dim, 'stride': microtile_m, 'wrap': buffer_dimension[outer_dim] // microtile_m}
            )
        else:
            tile_traversal.append({'dimension': dim, 'stride': 1, 'wrap': buffer_dimension[dim]})
    offset = [0 for _ in buffer_dimension]
    offset[inner_dim] = port * out_slice
    return make_staging_descriptor(
        access='write',
        view=view,
        tiling_dimension=tiling_dimension,
        offset=offset,
        tile_traversal=tile_traversal,
        inner_dim=inner_dim,
        outer_dim=outer_dim,
        io_boundary_shape='real',
        io_tiling_dimension=io_tiling_dimension,
    )


def describe_family_rhs_staging(view: TensorView, microtiling, parallelism, port: int, buf_dims=None):
    """Staging descriptor for an RHS (weight) tensor.

    For matmul rhs: view.tile_outer = K slice per port, view.tile_inner = N slice per port.
    """
    microtile_k = int(microtiling.microtile_k)
    microtile_n = int(microtiling.microtile_n)
    # The rhs view encodes both K and N slices: outer dim = K, inner dim = N.
    k_slice = view.tile_outer
    raw_k = view.tile_raw_outer
    n_slice = view.tile_inner
    raw_n = view.tile_raw_inner
    perm = view.perm
    if perm is not None:
        allowed = [list(range(len(perm)))]
        if len(perm) >= 2:
            swap_last_two = list(range(len(perm)))
            swap_last_two[-2], swap_last_two[-1] = swap_last_two[-1], swap_last_two[-2]
            allowed.append(swap_last_two)
        if list(perm) not in allowed:
            raise ValueError(f'matmul RHS does not support io_view permutation {perm}.')
    buffer_dimension = ordered_view_shape(view, 'full') if buf_dims is None else [int(x) for x in buf_dims]
    inner_dim, outer_dim, traversal_dims = canonical_buffer_axes(view)
    io_tiling_dimension = ordered_view_shape(view, 'logical')
    io_tiling_dimension[inner_dim] = raw_n
    io_tiling_dimension[outer_dim] = raw_k
    tiling_dimension = [1 for _ in buffer_dimension]
    tiling_dimension[inner_dim] = microtile_n
    tiling_dimension[outer_dim] = microtile_k

    row = int(port) // int(parallelism.cas_length)
    col = int(port) % int(parallelism.cas_length)
    offset = [0 for _ in buffer_dimension]
    offset[inner_dim] = row * n_slice
    offset[outer_dim] = col * k_slice

    tile_traversal = [
        {'dimension': inner_dim, 'stride': microtile_n, 'wrap': max(1, n_slice // microtile_n)},
        {'dimension': outer_dim, 'stride': microtile_k, 'wrap': max(1, k_slice // microtile_k)},
    ]
    used = {outer_dim, inner_dim}
    for dim in traversal_dims:
        if dim in used:
            continue
        tile_traversal.append({'dimension': dim, 'stride': 1, 'wrap': buffer_dimension[dim]})

    return make_staging_descriptor(
        access='read',
        view=view,
        tiling_dimension=tiling_dimension,
        offset=offset,
        tile_traversal=tile_traversal,
        inner_dim=inner_dim,
        outer_dim=outer_dim,
        boundary_shape='logical',
        io_boundary_shape='logical',
        io_tiling_dimension=io_tiling_dimension,
        extras={
            'packing': 'mmul_rhs',
            'packing_microtile_k': microtile_k,
            'packing_microtile_n': microtile_n,
        },
    )


def validate_family_tile_contract(
    *,
    node_name: str,
    precision,
    parallelism,
    microtiling,
    lhs_view: TensorView,
    output_view: TensorView,
    bank_bytes: int,
    rhs_overhead_bytes: int = 0,
) -> None:
    tile_outer = lhs_view.compacted_tile_outer
    tile_inner_lhs = lhs_view.tile_inner
    tile_inner_rhs = output_view.tile_inner
    full_inner_lhs = lhs_view.full_inner
    full_inner_rhs = output_view.full_inner

    a_tile_bytes = int(tile_outer) * tile_inner_lhs * storage_bytes_for_spec(precision['lhs'])
    b_tile_bytes = tile_inner_lhs * tile_inner_rhs * storage_bytes_for_spec(precision['rhs'])
    c_tile_bytes = int(tile_outer) * tile_inner_rhs * storage_bytes_for_spec(precision['output'])
    if a_tile_bytes > bank_bytes:
        raise ValueError(f'{node_name}: A tile uses {a_tile_bytes}B, exceeds one {bank_bytes}B bank.')
    if b_tile_bytes + int(rhs_overhead_bytes) > bank_bytes:
        raise ValueError(
            f'{node_name}: B tile plus overhead uses '
            f'{b_tile_bytes + int(rhs_overhead_bytes)}B, exceeds one {bank_bytes}B bank.'
        )
    if c_tile_bytes > bank_bytes:
        raise ValueError(f'{node_name}: C tile uses {c_tile_bytes}B, exceeds one {bank_bytes}B bank.')

    if full_inner_lhs != tile_inner_lhs * int(parallelism.cas_length):
        raise ValueError(f'{node_name}: full_inner_lhs must equal tile_inner_lhs * cas_length.')
    if full_inner_rhs != tile_inner_rhs * int(parallelism.cas_num):
        raise ValueError(f'{node_name}: full_inner_rhs must equal tile_inner_rhs * cas_num.')
    if tile_inner_lhs % max(1, 2 * int(microtiling.microtile_k)) != 0:
        raise ValueError(f'{node_name}: tile_inner_lhs must be divisible by 2 * microtile_k.')
    if tile_inner_rhs % max(1, 2 * int(microtiling.microtile_n)) != 0:
        raise ValueError(f'{node_name}: tile_inner_rhs must be divisible by 2 * microtile_n.')
    if lhs_view.compacted_tile_outer % max(1, 2 * int(microtiling.microtile_m)) != 0:
        raise ValueError(f'{node_name}: tile_outer (lhs) must be divisible by 2 * microtile_m.')


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
