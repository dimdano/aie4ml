from __future__ import annotations

import math
from typing import Any, Dict, List

from ...utils import canonical_buffer_axes, make_staging_descriptor, ordered_view_shape
from ...utils.precision import storage_bytes_for_spec


def elementwise_vec_size(lhs_precision, device) -> int:
    """Return the AIE-ML vector lane count for elementwise kernels.

    AIE-ML vector registers are 512-bit (64 bytes).  The lane count is
    the number of elements that fit in one full register:
      int8 → 64, int16/bfloat16 → 32, int32/float → 16.
    """
    lhs_bytes = storage_bytes_for_spec(lhs_precision)
    return int(device.vector_bytes) // max(1, int(lhs_bytes))


def validate_elementwise_tile_contract(
    *, node_name: str, precision: Dict[str, Any], lhs_view, bank_bytes: int, vec_size: int
) -> None:
    slice_elements = int(math.prod(lhs_view.tile))
    full_inner = lhs_view.full_inner
    if full_inner % vec_size != 0:
        raise ValueError(
            f'{node_name}: inner dimension {full_inner} is not a multiple of vec_size {vec_size}; '
            'the resolver must align full_inner to vec_size before building the view.'
        )
    lhs_tile_bytes = slice_elements * storage_bytes_for_spec(precision['lhs'])
    rhs_tile_bytes = slice_elements * storage_bytes_for_spec(precision['rhs'])
    out_tile_bytes = slice_elements * storage_bytes_for_spec(precision['output'])
    if lhs_tile_bytes > bank_bytes:
        raise ValueError(f'{node_name}: lhs tile uses {lhs_tile_bytes}B, exceeds one {bank_bytes}B bank.')
    if rhs_tile_bytes > bank_bytes:
        raise ValueError(f'{node_name}: rhs tile uses {rhs_tile_bytes}B, exceeds one {bank_bytes}B bank.')
    if out_tile_bytes > bank_bytes:
        raise ValueError(f'{node_name}: output tile uses {out_tile_bytes}B, exceeds one {bank_bytes}B bank.')


def describe_elementwise_staging(view, port: int, access: str, contract: str, buf_dims=None):
    """Build an elementwise staging descriptor for an 'outer' or 'inner' partition contract."""
    rank = len(view.real)
    inner_dim, outer_dim, _ = canonical_buffer_axes(view)

    partition_dim = inner_dim if contract == 'inner' else outer_dim
    traverse_dim = outer_dim if contract == 'inner' else inner_dim

    raw_slice = ordered_view_shape(view, 'tile_raw')
    buffer_dimension = ordered_view_shape(view, 'full') if buf_dims is None else [int(x) for x in buf_dims]

    offset = [0 for _ in buffer_dimension]
    offset[partition_dim] = int(port) * int(raw_slice[partition_dim])

    tile_traversal: List[Dict[str, int]] = [
        {
            'dimension': partition_dim,
            'stride': int(raw_slice[partition_dim]),
            'wrap': 1,
        },
        {
            'dimension': traverse_dim,
            'stride': int(raw_slice[traverse_dim]),
            # 'inner' contract: the outer axis is traversed in full across the port's inner slice.
            # 'outer' contract: each port covers exactly one outer slice, so wrap=1.
            'wrap': (
                max(1, int(buffer_dimension[traverse_dim]) // max(1, int(raw_slice[traverse_dim])))
                if contract == 'inner'
                else 1
            ),
        },
    ]
    for dim in range(rank):
        if dim in (partition_dim, traverse_dim):
            continue
        tile_traversal.append({'dimension': dim, 'stride': 1, 'wrap': int(buffer_dimension[dim])})

    tiling_dim = list(raw_slice)
    for dim in range(rank):
        if dim not in (inner_dim, outer_dim):
            tiling_dim[dim] = 1

    return make_staging_descriptor(
        access=access,
        view=view,
        tiling_dimension=tiling_dim,
        offset=offset,
        tile_traversal=tile_traversal,
        inner_dim=inner_dim,
        outer_dim=outer_dim,
        slice_dim=partition_dim,
        boundary_shape='tile_raw' if access == 'read' else None,
        io_boundary_shape='real',
        io_tiling_dimension=raw_slice,
    )
