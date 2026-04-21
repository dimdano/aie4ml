from __future__ import annotations

import copy
from typing import Any, Dict, List


def graph_input_port_descs(entry, ctx, port_base: int) -> Dict[int, Dict[str, Any]]:
    consumer = entry.consumers[0].consumer
    inst = ctx.ir.execution.get(consumer.name)
    count = int(inst.ports.inputs[entry.tensor].count)
    if count != int(entry.producer_ports):
        raise ValueError(
            f'{entry.tensor}: graph-input producer_ports must match consumer port count '
            f'({entry.producer_ports} != {count}).'
        )
    descs: Dict[int, Dict[str, Any]] = {}
    for port in range(count):
        descs[int(port_base) + int(port)] = inst.variant.describe_input_staging(
            consumer, inst.config, entry.tensor, port, None, None
        )
    return descs


def graph_input_full_descriptor(entry, ctx) -> Dict[str, Any]:
    consumer = entry.consumers[0].consumer
    inst = ctx.ir.execution.get(consumer.name)
    base = inst.variant.describe_input_staging(consumer, inst.config, entry.tensor, 0, None, None)
    io_tile = list(base['io_tiling_dimension'])
    return {
        'access': 'write',
        'buffer_dimension': list(base['buffer_dimension']),
        'tiling_dimension': list(io_tile),
        'io_tiling_dimension': list(io_tile),
        'io_boundary_dimension': list(base['io_boundary_dimension']),
        'offset': [0 for _ in io_tile],
        'slice_dimension': int(base['slice_dimension']),
        'inner_dimension': int(base['inner_dimension']),
        'outer_dimension': int(base['outer_dimension']),
    }


def graph_input_writer_port_descs(read_descs: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for port, base in read_descs.items():
        io_tile = list(base['io_tiling_dimension'])
        out[int(port)] = {
            'access': 'write',
            'buffer_dimension': list(base['buffer_dimension']),
            'tiling_dimension': list(io_tile),
            'io_tiling_dimension': list(io_tile),
            'io_boundary_dimension': list(base['io_boundary_dimension']),
            'offset': list(base['offset']),
            'slice_dimension': int(base['slice_dimension']),
            'inner_dimension': int(base['inner_dimension']),
            'outer_dimension': int(base['outer_dimension']),
        }
    return out


def graph_input_unit_box(descs: Dict[int, Dict[str, Any]], ports: List[int]):
    if not ports:
        raise ValueError('graph-input shard unit cannot be empty.')
    first = descs[int(ports[0])]
    rank = len(first['offset'])
    base = [None for _ in range(rank)]
    limit = [None for _ in range(rank)]
    for port in ports:
        desc = descs[int(port)]
        if len(desc['offset']) != rank:
            raise ValueError('graph-input port descriptors have inconsistent rank.')
        tile = list(desc['io_tiling_dimension'])
        offset = list(desc['offset'])
        for dim in range(rank):
            start = int(offset[dim])
            end = start + int(tile[dim])
            base[dim] = start if base[dim] is None else min(int(base[dim]), start)
            limit[dim] = end if limit[dim] is None else max(int(limit[dim]), end)
    return [int(v) for v in base], [int(limit[d] - base[d]) for d in range(rank)]


def graph_input_port_descriptor(entry, port: int) -> Dict[str, Any]:
    try:
        return copy.deepcopy(entry.graph_input_port_descs[int(port)])
    except KeyError as exc:
        raise RuntimeError(f'{entry.tensor}: missing graph-input descriptor for port {port}.') from exc


def graph_input_writer_port_descriptor(entry, port: int) -> Dict[str, Any]:
    try:
        return copy.deepcopy(entry.graph_input_writer_port_descs[int(port)])
    except KeyError as exc:
        raise RuntimeError(f'{entry.tensor}: missing graph-input writer descriptor for port {port}.') from exc


def localize_graph_io_descriptor(base: Dict[str, Any], unit_base: List[int], buf_dims: List[int]) -> Dict[str, Any]:
    desc = copy.deepcopy(base)
    offset = list(desc['offset'])
    boundary = list(desc.get('io_boundary_dimension', desc.get('boundary_dimension', buf_dims)))
    if len(offset) != len(unit_base) or len(offset) != len(buf_dims):
        raise RuntimeError('graph-input descriptor rank mismatch during localization.')
    for dim in range(len(offset)):
        offset[dim] -= int(unit_base[dim])
        boundary[dim] = min(int(buf_dims[dim]), max(0, int(boundary[dim]) - int(unit_base[dim])))
    desc['buffer_dimension'] = list(buf_dims)
    desc['offset'] = offset
    desc['boundary_dimension'] = boundary
    return desc
