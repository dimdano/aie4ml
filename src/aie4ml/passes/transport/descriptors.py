from __future__ import annotations

import copy
from typing import Any, Dict, MutableMapping, Sequence

from ...op_impls.utils import STORAGE_LAYOUT_LINEAR, STORAGE_LAYOUT_MICROTILED, STORAGE_LAYOUTS


def boundary_access_descriptor(base: Dict[str, Any], *, project_to_io_boundary: bool = False) -> Dict[str, Any]:
    """Lower a staging layout to an ADF graph-boundary access descriptor."""

    storage_layout = base.get('storage_layout')
    if storage_layout not in STORAGE_LAYOUTS:
        raise ValueError(f'Unknown or missing staging storage_layout {storage_layout!r}.')
    if storage_layout == STORAGE_LAYOUT_LINEAR:
        return copy.deepcopy(base)
    if storage_layout != STORAGE_LAYOUT_MICROTILED:
        raise NotImplementedError(f'No boundary DMA lowering for storage_layout {storage_layout!r}.')

    dimensions = [int(value) for value in base['buffer_dimension']]
    tile = [int(value) for value in base['tiling_dimension']]
    if len(dimensions) != 2 or len(tile) != 2:
        raise NotImplementedError('Microtiled boundary DMA requires a 2-D kernel buffer.')

    inner = int(base['inner_dimension'])
    outer = int(base['outer_dimension'])
    microtile_inner = tile[inner]
    microtile_outer = tile[outer]
    if inner != 0 or outer != 1 or any(int(value) != 0 for value in base['offset']):
        raise NotImplementedError('Microtiled boundary DMA requires a local, inner-contiguous buffer.')
    if microtile_outer == 1:
        return copy.deepcopy(base)

    inner_extent, outer_extent = dimensions
    if inner_extent % microtile_inner or outer_extent % microtile_outer:
        raise ValueError(
            f'Direct boundary shape {(outer_extent, inner_extent)} '
            f'is not divisible by microtile {(microtile_outer, microtile_inner)}.'
        )

    transfer_inner_extent, transfer_outer_extent = inner_extent, outer_extent
    if project_to_io_boundary:
        io_boundary = [int(value) for value in base['io_boundary_dimension']]
        if len(io_boundary) != 2:
            raise NotImplementedError('Direct boundary DMA projection requires a 2-D IO boundary.')
        transfer_inner_extent, transfer_outer_extent = io_boundary
        if transfer_inner_extent > inner_extent or transfer_outer_extent > outer_extent:
            raise ValueError(
                f'Direct boundary shape {(outer_extent, inner_extent)} cannot contain IO boundary '
                f'{(transfer_outer_extent, transfer_inner_extent)}.'
            )
        if transfer_inner_extent % microtile_inner or transfer_outer_extent % microtile_outer:
            raise NotImplementedError(
                f'Direct boundary DMA cannot project IO boundary '
                f'{(transfer_outer_extent, transfer_inner_extent)} '
                f'from microtile {(microtile_outer, microtile_inner)}.'
            )

    microtile_elements = microtile_outer * microtile_inner
    descriptor = copy.deepcopy(base)
    descriptor['buffer_dimension'] = [
        microtile_elements,
        inner_extent // microtile_inner,
        outer_extent // microtile_outer,
    ]
    descriptor['tiling_dimension'] = [microtile_inner, 1, 1]
    descriptor['offset'] = [0, 0, 0]
    descriptor['tile_traversal'] = [
        {'dimension': 1, 'stride': 1, 'wrap': transfer_inner_extent // microtile_inner},
        {'dimension': 0, 'stride': microtile_inner, 'wrap': microtile_outer},
        {'dimension': 2, 'stride': 1, 'wrap': transfer_outer_extent // microtile_outer},
    ]
    descriptor.pop('boundary_dimension', None)
    return descriptor


def rebase_descriptor_offset(descriptor: MutableMapping[str, Any], offset_base: Sequence[int]) -> None:
    """Rebase only a descriptor offset into endpoint-local coordinates."""
    if not offset_base:
        return
    offset = [int(value) for value in descriptor['offset']]
    base = [int(value) for value in offset_base]
    if len(offset) != len(base):
        raise RuntimeError(f'descriptor rank mismatch during offset rebasing ({len(offset)} != {len(base)}).')
    descriptor['offset'] = [offset[dim] - base[dim] for dim in range(len(offset))]


def localize_descriptor(
    descriptor: MutableMapping[str, Any],
    offset_base: Sequence[int],
    buffer_dimension: Sequence[int],
) -> None:
    """Rebase a descriptor and its boundaries into an endpoint-local buffer."""
    if not offset_base:
        return

    dims = [int(value) for value in buffer_dimension]
    rebase_descriptor_offset(descriptor, offset_base)
    base = [int(value) for value in offset_base]
    if len(dims) != len(base):
        raise RuntimeError(f'descriptor rank mismatch during localization ({len(dims)} != {len(base)}).')

    descriptor['buffer_dimension'] = list(dims)
    for key in ('boundary_dimension', 'io_boundary_dimension'):
        if key not in descriptor:
            continue
        boundary = [int(value) for value in descriptor[key]]
        if len(boundary) != len(dims):
            raise RuntimeError(f'descriptor {key} rank mismatch during localization.')
        descriptor[key] = [min(dims[dim], max(0, boundary[dim] - base[dim])) for dim in range(len(dims))]


def localized_graph_io_descriptor(
    descriptor: Dict[str, Any],
    offset_base: Sequence[int],
    buffer_dimension: Sequence[int],
) -> Dict[str, Any]:
    """Return a graph-IO descriptor localized to one memory-tile shard."""
    localized = copy.deepcopy(descriptor)
    dims = [int(value) for value in buffer_dimension]
    boundary = list(localized['io_boundary_dimension'])
    base = [int(value) for value in offset_base]
    if len(base) != len(dims) or len(boundary) != len(dims):
        raise RuntimeError('graph-IO descriptor rank mismatch during localization.')
    rebase_descriptor_offset(localized, base)
    localized['buffer_dimension'] = list(dims)
    if localized.get('access') == 'read':
        localized['boundary_dimension'] = [
            min(dims[dim], max(0, int(boundary[dim]) - base[dim])) for dim in range(len(dims))
        ]
    else:
        localized.pop('boundary_dimension', None)
    return localized
