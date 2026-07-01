from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence


@dataclass(frozen=True)
class TensorView:
    """Per-tensor shape contract for the execution/buffer view of one kernel instance.

    2-D buffer model:
      inner dim = kernel-facing axis [-1]
      outer dim = outer/work axis [-2]

    logical      LogicalIR TensorVar shape.
    real         Rank-preserving shape after io_view permutation.
    full         Rank-preserving aligned/padded execution shape.
    tile         Rank-preserving per-port aligned slice shape.
    tile_raw     Rank-preserving per-port raw slice shape.
    buffer_order Axis permutation from logical to buffer layout.
    perm         Optional io_view permutation applied before buffer layout.
    """

    logical: tuple[int, ...]
    real: tuple[int, ...]
    full: tuple[int, ...]
    tile: tuple[int, ...]
    tile_raw: tuple[int, ...]
    buffer_order: tuple[int, ...]
    perm: tuple[int, ...] | None = None

    # ------------------------------------------------------------------
    # Convenience accessors for the generic 2-D execution/hardware model.
    # inner = kernel-facing axis (last logical axis, index [-1]).
    # outer = outer axis    (second-to-last axis, index [-2]).
    # Accessing outer on a rank-1 view is undefined (wraps to [-1]).
    # ------------------------------------------------------------------

    @property
    def rank(self) -> int:
        """Number of dimensions in the real view."""
        return len(self.real)

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

    @property
    def compacted_tile_raw_outer(self) -> int:
        """2D raw per-port kernel outer extent before padding after compacting non-inner dims."""
        return int(math.prod(self.tile_raw[:-1])) if self.rank >= 2 else 1


def ordered_view_shape(view: TensorView, kind: str) -> list[int]:
    shape = getattr(view, kind)
    return [int(shape[index]) for index in view.buffer_order]


def _shape_from_buffer_order(buffer_shape: Sequence[int], buffer_order: Sequence[int]) -> tuple[int, ...]:
    if len(buffer_shape) != len(buffer_order):
        raise ValueError(f'buffer shape rank {len(buffer_shape)} does not match buffer_order rank {len(buffer_order)}.')
    shape = [0 for _ in buffer_order]
    for buffer_dim, view_axis in enumerate(buffer_order):
        shape[int(view_axis)] = int(buffer_shape[buffer_dim])
    return tuple(shape)


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
    rank = len(view.real)
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
    # buffer_dimension: full backing-buffer size (= view.full in buffer order).
    # tiling_dimension: AMD ADF DMA transfer-chunk size per BD transaction.
    #   This is AMD's own term — NOT the same as TensorView.tile (per-kernel slice).
    # inner_dimension / outer_dimension: aie4ml axis tags for downstream passes.
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
    real = tuple(int(x) for x in view_shape(node, tensor, direction))
    layout = view_layout(node, tensor, direction)
    rank = len(real)

    full = list(real)
    full[-1] = int(full_inner)
    if rank >= 2:
        full[-2] = int(full_outer)

    tile = list(full)
    tile[-1] = int(tile_inner)
    if tile_outer is not None and rank >= 2:
        tile[-2] = int(tile_outer)

    tile_raw = list(real)
    tile_raw[-1] = int(tile_inner_raw)
    if tile_outer_raw is not None and rank >= 2:
        tile_raw[-2] = int(tile_outer_raw)
    elif tile_outer is not None and rank >= 2:
        tile_raw[-2] = int(tile_outer)

    return TensorView(
        logical=logical,
        real=real,
        full=tuple(full),
        tile=tuple(tile),
        tile_raw=tuple(tile_raw),
        buffer_order=tuple(int(x) for x in layout['buffer_order']),
        perm=None if layout.get('perm') is None else tuple(int(x) for x in layout['perm']),
    )


def build_tensor_view_from_staging(node, tensor, direction: str, desc: Mapping[str, Any]) -> TensorView:
    """Build a TensorView whose per-port tile matches an inherited staging descriptor."""

    from .io import view_layout, view_shape

    logical = tuple(int(x) for x in tensor.shape)
    real = tuple(int(x) for x in view_shape(node, tensor, direction))
    layout = view_layout(node, tensor, direction)
    buffer_order = tuple(int(x) for x in layout['buffer_order'])
    full = _shape_from_buffer_order(desc['buffer_dimension'], buffer_order)
    tile = _shape_from_buffer_order(_staging_tile_shape(desc), buffer_order)

    return TensorView(
        logical=logical,
        real=real,
        full=full,
        tile=tile,
        tile_raw=tile,
        buffer_order=buffer_order,
        perm=None if layout.get('perm') is None else tuple(int(x) for x in layout['perm']),
    )
