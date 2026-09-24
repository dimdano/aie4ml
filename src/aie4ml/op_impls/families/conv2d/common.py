"""What Conv2D's semantic contract and its kernels share: the channel-block frame vocabulary."""

from __future__ import annotations

from ....ir.graph import OpNode
from ...utils import (
    STORAGE_LAYOUT_INNER_BLOCKED,
    STORAGE_LAYOUT_LINEAR,
    AxisPlan,
    SpatialAccess2D,
    TensorView,
    build_padded_spatial_view,
    build_staging_descriptor,
    canonical_buffer_axes,
    ordered_view_shape,
    shared_consumer_spatial_access,
)

CHANNEL_BLOCK = 8
"""Channels per mmul K/N block: the frame's inner blocking factor, and the partition granularity."""

ROW_ALIGN_PIXELS = 4
"""Frame column granularity that keeps every int8 frame row 32-byte aligned."""


def spatial_access_of(node: OpNode) -> SpatialAccess2D:
    """The window a conv2d node reads, in the generic spatial vocabulary."""
    return SpatialAccess2D(
        kernel=tuple(int(x) for x in node.metadata['kernel_shape']),
        pads=tuple(int(p) for p in node.metadata['pads']),
        strides=tuple(int(s) for s in node.metadata['strides']),
        dilations=tuple(int(d) for d in node.metadata['dilations']),
    )


def frame_view(tensor, *, column_block: int, column_align: int, channel_slices: int = 1, bands: int = 1) -> TensorView:
    """The padded frame of one activation, as every op on either side of it sees it."""
    return build_padded_spatial_view(
        tensor.shape,
        shared_consumer_spatial_access(tensor),
        column_block=column_block,
        column_align=column_align,
        inner_block=CHANNEL_BLOCK,
        inner_slices=channel_slices,
        row_slices=bands,
        row_bytes_align=ROW_ALIGN_PIXELS,
    )


def describe_logical_staging(view: TensorView, access: str, *, transfer_bytes: int = 0, rows=None):
    """Staging of an activation carried as the logical tensor, in its own order.

    A stream carries a wire order, not a memory layout, and a retiled buffer port receives what the
    boundary carries; either way that is the tensor itself -- rows, then columns, then channels --
    and the padding the kernel computes with (a zero border, channels rounded to a block, a width
    rounded to whole register tiles) is built on the tile, where it belongs. So this publishes the
    logical window, not the execution frame.

    `rows` = (first, count) limits the port to those rows of the tensor, still in order: one row band's
    window. `transfer_bytes` is what one inference moves when that differs from the tensor: a buffer is
    whole 16-byte beats, so a tensor that is not is followed by zero padding that is not part of it.
    """
    # The descriptor is built on a view of the tensor: the execution frame is the kernel's business
    # and never reaches the port.
    wire = TensorView(logical=view.logical, full=view.logical, tile=view.logical, tile_raw=view.logical)
    inner_dim, _outer_dim, traversal_dims = canonical_buffer_axes(wire)
    shape = ordered_view_shape(wire, 'logical')
    plans = {dim: AxisPlan(int(shape[dim]), int(shape[dim]), 1) for dim in traversal_dims}
    origin = {dim: 0 for dim in range(wire.rank)}
    io_rows = {}
    if rows is not None:
        first, count = (int(x) for x in rows)
        row_dim = wire.buffer_order.index(1)
        plans[row_dim] = AxisPlan(count, count, 1, first)
        origin[row_dim] = first
        io_rows[row_dim] = count
    extras = {'storage_layout': STORAGE_LAYOUT_LINEAR}
    if transfer_bytes:
        extras['transfer_bytes'] = int(transfer_bytes)
    return build_staging_descriptor(
        wire,
        access=access,
        plans=plans,
        order=traversal_dims,
        io_tiling_base='logical',
        io_tiling_overrides=io_rows,
        boundary_shape='logical' if access == 'read' else None,
        slice_dim=inner_dim,
        logical_origin=origin,
        extras=extras,
    )


def describe_frame_staging(
    view: TensorView, access: str, port: int, *, band: int = 0, band_rows: int = 0, column_phases: int = 1
):
    """Staging of one port's window on a spatial frame.

    The frame holds `CHANNEL_BLOCK` channels per chunk with the chunk index outermost, so a port's
    share of the channels is a contiguous region -- which is what makes the channel axis the
    partition axis for both the cascade split and the chain split. A row band is the other
    partition: it starts `band * band_rows` into the frame and runs for the tile's rows, which
    overlap the neighbouring bands by the window span.

    `column_phases` > 1 marks a frame whose columns are grouped by their residue modulo it, as a
    strided window reads them. Only a retiler writes one, and the marker keeps any frame in plain
    column order from ever matching it.
    """
    extras = {'storage_layout': STORAGE_LAYOUT_INNER_BLOCKED}
    if column_phases > 1:
        extras['column_phases'] = int(column_phases)
    inner_dim, _outer_dim, traversal_dims = canonical_buffer_axes(view)
    row_dim = view.buffer_order.index(1)
    blocks = int(view.tile[-1]) // CHANNEL_BLOCK
    origin = ordered_view_shape(view, 'origin')
    tile = ordered_view_shape(view, 'tile')
    row_offset = int(band) * int(band_rows)
    plans = {inner_dim: AxisPlan(CHANNEL_BLOCK, CHANNEL_BLOCK, blocks, int(port) * blocks * CHANNEL_BLOCK)}
    if band_rows:
        plans[row_dim] = AxisPlan(int(tile[row_dim]), int(tile[row_dim]), 1, row_offset)
    # The frame is the image inside its zero border, so a window starts `origin` before the image.
    starts = {dim: 0 for dim in range(view.rank)}
    starts[inner_dim] = int(port) * blocks * CHANNEL_BLOCK
    starts[row_dim] = row_offset
    return build_staging_descriptor(
        view,
        access=access,
        plans=plans,
        order=traversal_dims,
        io_tiling_base='tile',
        logical_origin={dim: starts[dim] - int(origin[dim]) for dim in range(view.rank)},
        boundary_shape='logical' if access == 'read' else None,
        extras=extras,
    )
