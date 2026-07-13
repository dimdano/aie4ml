from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, NamedTuple, Sequence


@dataclass(frozen=True)
class TensorView:
    """Per-tensor shape contract across three axis spaces:

      LOGICAL  host order (TensorVar shape); host-facing io_tiling/io_boundary live here.
      VIEW     kernel order = LOGICAL after `perm`; inner axis [-1], outer [-2].
               full/tile/tile_raw live here.
      BUFFER   memory order = LOGICAL reversed; BD fields (buffer_dimension,
               tiling_dimension, offset, tile_traversal) live here.

    `buffer_order` lists LOGICAL axes, so a VIEW-order shape relabels to LOGICAL before
    it applies (see `ordered_view_shape`). Identity `perm` collapses all three spaces,
    so the distinction only surfaces under transpose.

    logical             unpadded, LOGICAL order (the only non-derivable shape).
    full/tile/tile_raw  padded / per-port slice / raw slice, VIEW order.
    perm                io_view permutation (VIEW axis -> LOGICAL axis), or None.
    """

    logical: tuple[int, ...]
    full: tuple[int, ...]
    tile: tuple[int, ...]
    tile_raw: tuple[int, ...]
    perm: tuple[int, ...] | None = None

    # Convenience accessors for the generic 2-D execution/hardware model.
    # inner = last axis [-1], outer = [-2] (1 on a rank-1 view).

    @property
    def rank(self) -> int:
        return len(self.logical)

    @property
    def buffer_order(self) -> tuple[int, ...]:
        """LOGICAL axis per buffer position — the fixed axis-reversal convention."""
        return tuple(reversed(range(self.rank)))

    @property
    def full_inner(self) -> int:
        """Fully padded inner-dim (last axis) extent."""
        return int(self.full[-1])

    @property
    def full_outer(self) -> int:
        """Fully padded outer-dim (second-to-last axis) extent."""
        return int(self.full[-2]) if self.rank >= 2 else 1

    @property
    def compacted_full_outer(self) -> int:
        """2D kernel-contract outer extent after compacting all non-inner dims."""
        return int(math.prod(self.full[:-1])) if self.rank >= 2 else 1

    @property
    def tile_inner(self) -> int:
        """Per-port tile extent along the inner dim."""
        return int(self.tile[-1])

    @property
    def tile_outer(self) -> int:
        """Per-port tile extent along the outer dim."""
        return int(self.tile[-2]) if self.rank >= 2 else 1

    @property
    def compacted_tile_outer(self) -> int:
        """2D per-port kernel outer extent after compacting all non-inner dims."""
        return int(math.prod(self.tile[:-1])) if self.rank >= 2 else 1

    @property
    def tile_raw_inner(self) -> int:
        """Per-port unaligned tile extent along the inner dim."""
        return int(self.tile_raw[-1])

    @property
    def tile_raw_outer(self) -> int:
        """Per-port unaligned tile extent along the outer dim."""
        return int(self.tile_raw[-2]) if self.rank >= 2 else 1


def _view_to_logical(shape: Sequence[int], perm: Sequence[int] | None) -> tuple[int, ...]:
    """Relabel a VIEW-order shape (full/tile/tile_raw) to LOGICAL order."""
    if perm is None:
        return tuple(int(x) for x in shape)
    relabelled = [0 for _ in shape]
    for view_axis, logical_axis in enumerate(perm):
        relabelled[int(logical_axis)] = int(shape[view_axis])
    return tuple(relabelled)


def _logical_to_view(shape: Sequence[int], perm: Sequence[int] | None) -> tuple[int, ...]:
    """Relabel a LOGICAL-order shape to VIEW order (inverse of `_view_to_logical`)."""
    if perm is None:
        return tuple(int(x) for x in shape)
    return tuple(int(shape[int(logical_axis)]) for logical_axis in perm)


def ordered_view_shape(view: TensorView, kind: str) -> list[int]:
    """`kind` field in BUFFER order. VIEW-order kinds (full/tile/tile_raw) relabel to
    LOGICAL first, since `buffer_order` lists LOGICAL axes; 'logical' is already there.
    """
    shape = getattr(view, kind)
    if kind != 'logical':
        shape = _view_to_logical(shape, view.perm)
    return [int(shape[index]) for index in view.buffer_order]


def _staging_tile_shape(desc: Mapping[str, Any]) -> tuple[int, ...]:
    if 'tiling_dimension' not in desc:
        raise ValueError('staging descriptor is missing tiling_dimension.')
    shape = [int(x) for x in desc['tiling_dimension']]
    for entry in desc.get('tile_traversal', ()):
        dim = int(entry['dimension'])
        shape[dim] = int(entry['stride']) * int(entry['wrap'])
    return tuple(shape)


def map_view_axis(view: TensorView, axis: int) -> int:
    if view.perm is None:
        return int(axis)
    return int(view.perm[axis])


def canonical_buffer_axes(view: TensorView) -> tuple[int, int, list[int]]:
    """Return (inner_dim, outer_dim, traversal_order) in buffer space.

    inner_dim       buffer axis for the last logical axis (inner / kernel-vectorized).
    outer_dim       buffer axis for the second-to-last logical axis (outer / work).
    traversal_order [inner_dim, outer_dim] + remaining axes sorted.
    """
    rank = view.rank
    inner_axis = map_view_axis(view, rank - 1)
    outer_axis = map_view_axis(view, rank - 2)
    inner_dim = view.buffer_order.index(int(inner_axis))
    outer_dim = view.buffer_order.index(int(outer_axis))
    tail_dims = sorted(dim for dim in range(rank) if dim not in (inner_dim, outer_dim))
    return inner_dim, outer_dim, [inner_dim, outer_dim] + tail_dims


def make_staging_descriptor(
    *,
    access: str,
    view: TensorView,
    tiling_dimension: Sequence[int],
    offset: Sequence[int],
    tile_traversal: Sequence[Mapping[str, int]],
    inner_dim: int,
    outer_dim: int,
    slice_dim: int | None = None,
    boundary_shape: str | None = None,
    io_boundary_shape: str | None = None,
    io_tiling_dimension: Sequence[int] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    # Axis space (see TensorView): BUFFER order = buffer_dimension, tiling_dimension,
    # offset, tile_traversal, inner/outer/slice_dimension (the BD walk — must agree).
    # LOGICAL order = io_tiling_dimension, io_boundary_dimension (host-facing).
    # tiling_dimension is ADF's per-BD chunk, NOT TensorView.tile.
    descriptor: Dict[str, Any] = {
        'access': access,
        'buffer_dimension': ordered_view_shape(view, 'full'),
        'tiling_dimension': [int(x) for x in tiling_dimension],
        'offset': [int(x) for x in offset],
        'tile_traversal': [{str(k): int(v) for k, v in entry.items()} for entry in tile_traversal],
        'slice_dimension': int(inner_dim if slice_dim is None else slice_dim),
        'inner_dimension': int(inner_dim),
        'outer_dimension': int(outer_dim),
    }

    if boundary_shape is not None:
        descriptor['boundary_dimension'] = ordered_view_shape(view, boundary_shape)
    if io_boundary_shape is not None:
        descriptor['io_boundary_dimension'] = ordered_view_shape(view, io_boundary_shape)
    if io_tiling_dimension is not None:
        descriptor['io_tiling_dimension'] = [int(x) for x in io_tiling_dimension]
    if extras:
        descriptor.update(dict(extras))
    return descriptor


class AxisPlan(NamedTuple):
    """DMA plan for one BUFFER axis: chunk = per-BD transfer size (tiling_dimension),
    stride/wrap = traversal, offset = port window shift."""

    chunk: int
    stride: int
    wrap: int
    offset: int = 0


def build_staging_descriptor(
    view: TensorView,
    *,
    access: str,
    plans: Mapping[int, AxisPlan],
    order: Sequence[int],
    io_tiling_base: str = 'logical',
    io_tiling_overrides: Mapping[int, int] | None = None,
    buf_dims: Sequence[int] | None = None,
    boundary_shape: str | None = None,
    io_boundary_shape: str | None = 'logical',
    slice_dim: int | None = None,
    extras: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Single BD-staging emitter: owns the BUFFER-order arrays and the LOGICAL-order
    io_tiling_dimension so no family hand-rolls the axis-space projection.

    `plans` maps a BUFFER axis -> AxisPlan; `order` lists all BUFFER axes in traversal
    order (axes absent from `plans` stream the whole extent, chunk 1). `io_tiling_base`
    seeds io_tiling_dimension; `io_tiling_overrides` set per-axis raw extents.
    """
    buffer_dimension = ordered_view_shape(view, 'full') if buf_dims is None else [int(x) for x in buf_dims]
    inner_dim, outer_dim, _ = canonical_buffer_axes(view)

    io_tiling_dimension = ordered_view_shape(view, io_tiling_base)
    for dim, raw in (io_tiling_overrides or {}).items():
        io_tiling_dimension[int(dim)] = int(raw)

    tiling_dimension = [1 for _ in buffer_dimension]
    offset = [0 for _ in buffer_dimension]
    tile_traversal = []
    for dim in order:
        plan = plans.get(int(dim))
        if plan is None:
            tile_traversal.append({'dimension': int(dim), 'stride': 1, 'wrap': int(buffer_dimension[dim])})
            continue
        tiling_dimension[int(dim)] = int(plan.chunk)
        offset[int(dim)] = int(plan.offset)
        tile_traversal.append({'dimension': int(dim), 'stride': int(plan.stride), 'wrap': int(plan.wrap)})

    return make_staging_descriptor(
        access=access,
        view=view,
        tiling_dimension=tiling_dimension,
        offset=offset,
        tile_traversal=tile_traversal,
        inner_dim=inner_dim,
        outer_dim=outer_dim,
        slice_dim=slice_dim,
        boundary_shape=boundary_shape,
        io_boundary_shape=io_boundary_shape,
        io_tiling_dimension=io_tiling_dimension,
        extras=extras,
    )


def describe_partition_staging(view, port: int, access: str, contract: str, buf_dims=None):
    """Partition pattern: split one axis across ports, stream the rest. Shared by
    row-wise/reduction ops. `contract` picks the partitioned axis:
    'inner' the kernel axis,'outer' the work axis.
    """
    inner_dim, outer_dim, _ = canonical_buffer_axes(view)
    is_inner = contract == 'inner'
    partition_dim = inner_dim if is_inner else outer_dim
    traverse_dim = outer_dim if is_inner else inner_dim

    part_raw = view.tile_raw_inner if is_inner else view.tile_raw_outer
    trav_raw = view.tile_raw_outer if is_inner else view.tile_raw_inner
    trav_full = view.full_outer if is_inner else view.full_inner
    traverse_wrap = max(1, int(trav_full) // max(1, int(trav_raw))) if is_inner else 1

    order = [partition_dim, traverse_dim] + [d for d in range(view.rank) if d not in (partition_dim, traverse_dim)]

    return build_staging_descriptor(
        view,
        access=access,
        plans={
            partition_dim: AxisPlan(int(part_raw), int(part_raw), 1, int(port) * int(part_raw)),
            traverse_dim: AxisPlan(int(trav_raw), int(trav_raw), int(traverse_wrap)),
        },
        order=order,
        io_tiling_base='tile_raw',
        slice_dim=partition_dim,
        buf_dims=buf_dims,
        boundary_shape='logical' if access == 'read' else None,
    )


def build_tensor_view(
    node,
    tensor,
    direction: str,
    *,
    full_inner: int,
    tile_inner: int,
    tile_inner_raw: int,
    full_outer: int,
    tile_outer: int | None = None,
    tile_outer_raw: int | None = None,
) -> TensorView:
    """Build a rank-preserving TensorView from resolved IO layout and per-port extents.

    Only the explicit inner axis and, for rank >= 2, the explicit outer axis are
    rewritten here. Compacted 2D kernel extents are derived later from the rank-preserving view.
    """

    from .io import view_layout, view_shape  # local import avoids circular dependency

    logical = tuple(int(x) for x in tensor.shape)
    # view-order base shape (= logical under perm); seeds the padded/sliced shapes.
    view = tuple(int(x) for x in view_shape(node, tensor, direction))
    layout = view_layout(node, tensor, direction)
    rank = len(view)

    full = list(view)
    full[-1] = int(full_inner)
    if rank >= 2:
        full[-2] = int(full_outer)

    tile = list(full)
    tile[-1] = int(tile_inner)
    if tile_outer is not None and rank >= 2:
        tile[-2] = int(tile_outer)

    tile_raw = list(view)
    tile_raw[-1] = int(tile_inner_raw)
    if tile_outer_raw is not None and rank >= 2:
        tile_raw[-2] = int(tile_outer_raw)
    elif tile_outer is not None and rank >= 2:
        tile_raw[-2] = int(tile_outer)

    return TensorView(
        logical=logical,
        full=tuple(full),
        tile=tuple(tile),
        tile_raw=tuple(tile_raw),
        perm=None if layout.get('perm') is None else tuple(int(x) for x in layout['perm']),
    )


def build_tensor_view_from_staging(node, tensor, direction: str, desc: Mapping[str, Any]) -> TensorView:
    """Build a TensorView whose per-port tile matches an inherited staging descriptor."""

    from .io import view_layout

    logical = tuple(int(x) for x in tensor.shape)
    layout = view_layout(node, tensor, direction)
    perm = None if layout.get('perm') is None else tuple(int(x) for x in layout['perm'])
    # BUFFER order is LOGICAL reversed; undo it, then relabel LOGICAL -> VIEW.
    full = _logical_to_view(tuple(reversed(desc['buffer_dimension'])), perm)
    tile = _logical_to_view(tuple(reversed(_staging_tile_shape(desc))), perm)

    return TensorView(
        logical=logical,
        full=full,
        tile=tile,
        tile_raw=tile,
        perm=perm,
    )
