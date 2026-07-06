from __future__ import annotations

from typing import Dict, List

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
        boundary_shape='real' if access == 'read' else None,
        io_boundary_shape='real',
        io_tiling_dimension=raw_slice,
    )
