from __future__ import annotations

import copy
from typing import Any, Dict, List

from ...op_impls.utils import STORAGE_LAYOUT_LINEAR
from .descriptors import rebase_descriptor_offset


def graph_input_port_descs(entry, ctx, port_base: int) -> Dict[int, Dict[str, Any]]:
    consumer = entry.single_consumer()
    inst = ctx.ir.execution.get(consumer.node.name)
    consumer_ports = consumer.selected_ports(inst.ports.inputs[consumer.tensor].count)
    count = len(consumer_ports)
    if count != int(entry.producer_port_count):
        raise ValueError(
            f'{entry.logical_tensor}: graph-input producer_ports must match consumer port count '
            f'({entry.producer_port_count} != {count}).'
        )
    descs: Dict[int, Dict[str, Any]] = {}
    for local_port, port in enumerate(consumer_ports):
        graph_port = int(port_base) + int(local_port)
        descs[graph_port] = inst.variant.describe_input_staging(
            consumer.node, inst.config, consumer.tensor, int(port), None, None
        )
        rebase_descriptor_offset(descs[graph_port], consumer.offset_base)
    return descs


def graph_input_full_descriptor(entry, ctx) -> Dict[str, Any]:
    consumer = entry.single_consumer()
    inst = ctx.ir.execution.get(consumer.node.name)
    port = int(consumer.selected_ports(inst.ports.inputs[consumer.tensor].count)[0])
    base = inst.variant.describe_input_staging(consumer.node, inst.config, consumer.tensor, port, None, None)
    rebase_descriptor_offset(base, consumer.offset_base)
    io_tile = list(base['io_tiling_dimension'])
    return {
        'access': 'write',
        'storage_layout': STORAGE_LAYOUT_LINEAR,
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
            'storage_layout': STORAGE_LAYOUT_LINEAR,
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
        return copy.deepcopy(entry.graph_input.port_descriptors[int(port)])
    except KeyError as exc:
        raise RuntimeError(f'{entry.logical_tensor}: missing graph-input descriptor for port {port}.') from exc


def graph_input_writer_port_descriptor(entry, port: int) -> Dict[str, Any]:
    try:
        return copy.deepcopy(entry.graph_input.writer_descriptors[int(port)])
    except KeyError as exc:
        raise RuntimeError(f'{entry.logical_tensor}: missing graph-input writer descriptor for port {port}.') from exc
