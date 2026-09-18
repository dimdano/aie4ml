# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from math import prod
from typing import Any, Dict, List, Optional

from ...aie_types import AIEDataType
from ...ir import get_backend_context, input_role
from ...op_impls.utils import staging_tile_shape
from ..base import AIEPass
from ..utils import sanitize_identifier
from .boundary import (
    graph_input_port_descriptor,
    graph_input_writer_port_descriptor,
)
from .collect import TransportCollector
from .descriptors import localize_descriptor, localized_graph_io_descriptor
from .model import EdgeEntry


class BuildMemoryPlan(AIEPass):
    def __init__(self):
        self.name = 'plan_memory'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        ctx.ir.physical.plan = _MemoryPlanMaterializer(ctx).build(list(ctx.ir.logical))
        return True


class CollectMemoryEntries(AIEPass):
    def __init__(self):
        self.name = 'collect_memory_entries'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        state = _MemoryPlanMaterializer(ctx).collect(list(ctx.ir.logical))
        ctx.ir.physical.plan = {'_memory_plan_state': state}
        return True


class MaterializeMemoryPlan(AIEPass):
    def __init__(self):
        self.name = 'materialize_memory_plan'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        planner = _MemoryPlanMaterializer(ctx)
        state = ctx.ir.physical.plan['_memory_plan_state']
        ctx.ir.physical.plan = planner.materialize(state)
        return True


class _MemoryPlanMaterializer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.device = ctx.device

        self.buffers = []
        self.direct_edges = []
        self.io_ports = []
        self.kernel_read_accesses = []
        self.kernel_write_accesses = []
        self.layer_indices = {}
        self._next_graph_output_port = 0
        self._max_graph_input_port = -1
        self._buffer_seq: Dict[str, int] = {}

    def build(self, nodes):
        state = self.collect(nodes)
        return self.materialize(_legalize_collected_entries(self.ctx, state))

    def collect(self, nodes):
        idx = 0
        for n in nodes:
            if self._kernel_inst(n):
                idx += 1
                self.layer_indices[n.name] = idx

        return {
            'layer_indices': dict(self.layer_indices),
            'entries': TransportCollector(self.ctx).collect(nodes),
        }

    def materialize(self, state):
        self.buffers = []
        self.direct_edges = []
        self.io_ports = []
        self.kernel_read_accesses = []
        self.kernel_write_accesses = []
        self.layer_indices = dict(state['layer_indices'])
        self._next_graph_output_port = 0
        self._max_graph_input_port = -1
        self._buffer_seq = {}

        for entry in state['entries']:
            self._materialize_entry(entry)

        return {
            'buffers': self.buffers,
            'direct_edges': self.direct_edges,
            'io_ports': self.io_ports,
            'kernel_read_accesses': self.kernel_read_accesses,
            'kernel_write_accesses': self.kernel_write_accesses,
            'graph_input_count': self._max_graph_input_port + 1,
            'graph_output_count': self._next_graph_output_port,
        }

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def _materialize_entry(self, entry: EdgeEntry):
        if len(entry.consumers) > 1:
            raise RuntimeError(f'{entry.logical_tensor}: materializer requires at most one consumer per entry.')
        if entry.consumers and entry.graph_output:
            raise RuntimeError(
                f'{entry.logical_tensor}: materializer does not accept mixed consumer + graph_output entry.'
            )
        if entry.unit is None:
            raise RuntimeError(f'{entry.logical_tensor}: missing transport unit; run memtile legalization first.')
        p_ports = [int(x) for x in entry.unit.producer_ports]
        c_ports = [int(x) for x in entry.unit.consumer_ports]
        realization = self._route(entry)
        if realization == 'direct':
            if entry.unit.count != 1:
                raise RuntimeError(f'{entry.logical_tensor}: direct realization invariant violated.')
            if entry.producer.node is None:
                if entry.graph_input is None or len(entry.consumers) != 1 or len(p_ports) != len(c_ports):
                    raise RuntimeError(f'{entry.logical_tensor}: direct graph-input invariant violated.')
                self._emit_direct_graph_input(entry, p_ports, c_ports)
            elif entry.graph_output:
                if entry.consumers or c_ports:
                    raise RuntimeError(f'{entry.logical_tensor}: direct graph-output invariant violated.')
                self._emit_direct_graph_output(entry, p_ports)
            elif len(entry.consumers) == 1 and len(p_ports) == len(c_ports):
                self._emit_direct_internal(entry, p_ports, c_ports)
            else:
                raise RuntimeError(f'{entry.logical_tensor}: direct internal-edge invariant violated.')
            return
        if realization != 'memtile':
            raise RuntimeError(f'{entry.logical_tensor}: unsupported transport realization {realization!r}.')
        graph_input_generic = entry.producer.node is None and entry.graph_input is not None
        if not graph_input_generic and (
            entry.unit.dimension is None or entry.unit.port_stride is None or entry.unit.dimension_size is None
        ):
            raise RuntimeError(f'{entry.logical_tensor}: missing shard metadata; run port-limit legalization first.')
        self._emit_memtile(entry, p_ports, c_ports)

    def _route(self, entry: EdgeEntry) -> str:
        if entry.decision is None:
            raise RuntimeError(f'{entry.logical_tensor}: missing transport decision; run classification first.')
        return entry.decision.realization

    # ------------------------------------------------------------------
    # Direct
    # ------------------------------------------------------------------

    def _emit_direct_internal(self, entry, p_ports, c_ports):
        p = entry.producer
        c = entry.single_consumer()

        for p_port, c_port in zip(p_ports, c_ports):
            self.direct_edges.append(
                {
                    'source': f'{sanitize_identifier(p.node.name)}.{p.group}[{int(p_port)}]',
                    'target': f'{sanitize_identifier(c.node.name)}.{c.group}[{int(c_port)}]',
                    'tensor': entry.logical_tensor,
                }
            )

    def _emit_direct_graph_input(self, entry, graph_ports, consumer_ports):
        consumer = entry.single_consumer()
        consumer_id = sanitize_identifier(consumer.node.name)
        inst = self._kernel_inst(consumer.node)
        dtype = self._graph_input_dtype(entry).to_dict()

        for graph_port, consumer_port in zip(graph_ports, consumer_ports):
            endpoint = f'{consumer_id}.{consumer.group}[{int(consumer_port)}]'
            staging = graph_input_writer_port_descriptor(entry, int(graph_port))
            descriptor = graph_input_port_descriptor(entry, int(graph_port))
            self._localize_direct_descriptor(descriptor)
            access_descriptor = descriptor
            if not self.device.has_memtile:
                access_descriptor = _logical_stream_microtiled_2d_descriptor(descriptor)
                if access_descriptor is not None:
                    descriptor = access_descriptor

            self.direct_edges.append(
                {
                    'source': f'ifm[{int(graph_port)}]',
                    'target': endpoint,
                    'tensor': entry.logical_tensor,
                }
            )
            # Vitis accepts access constraints on hierarchical ports but does not apply their buffer reorder;
            # bind read/write access to the concrete kk[...].in/out[...] port instead.
            access_endpoints = inst.variant.boundary_input_access_endpoints(
                inst.config, int(consumer_port), consumer.group
            )
            if not access_endpoints:
                raise RuntimeError(f'{entry.logical_tensor}: direct graph input has no concrete kernel endpoint.')
            if access_descriptor is not None:
                self.kernel_write_accesses.extend(
                    {'endpoint': f'{consumer_id}.{access_endpoint}', 'descriptor': access_descriptor}
                    for access_endpoint in access_endpoints
                )
            self.io_ports.append(
                {
                    'direction': 'input',
                    'port': int(graph_port),
                    'tensor': entry.logical_tensor,
                    'endpoint': endpoint,
                    'descriptor': descriptor,
                    'staging': staging,
                    'dtype': dtype,
                }
            )
            self._max_graph_input_port = max(self._max_graph_input_port, int(graph_port))

    def _emit_direct_graph_output(self, entry, producer_ports):
        producer = entry.producer
        producer_id = sanitize_identifier(producer.node.name)
        inst = self._kernel_inst(producer.node)
        dtype = self._graph_output_dtype(entry).to_dict()

        for producer_port in producer_ports:
            endpoint = f'{producer_id}.{producer.group}[{int(producer_port)}]'
            base = inst.variant.describe_output_staging(
                producer.node, inst.config, producer.tensor, int(producer_port), None
            )
            staging = _host_visible_output_staging(base)
            descriptor = dict(base)
            self._localize_direct_descriptor(descriptor)
            elements = int(prod(staging_tile_shape(descriptor)))
            logical_elements = int(prod(int(value) for value in staging['io_tiling_dimension']))
            if logical_elements != elements:
                raise NotImplementedError(
                    f'{entry.logical_tensor}: direct graph output requires {elements} kernel-buffer elements '
                    f'but exposes {logical_elements} logical elements; an output relayout adapter is required.'
                )
            access_descriptor = descriptor
            if not self.device.has_memtile:
                access_descriptor = _logical_stream_microtiled_2d_descriptor(descriptor)
                if access_descriptor is not None:
                    descriptor = access_descriptor
            graph_port = self._next_graph_output_port
            self._next_graph_output_port += 1

            self.direct_edges.append(
                {
                    'source': endpoint,
                    'target': f'ofm[{graph_port}]',
                    'tensor': entry.logical_tensor,
                }
            )
            access_endpoints = inst.variant.boundary_output_access_endpoints(inst.config, int(producer_port))
            if len(access_endpoints) != 1:
                raise RuntimeError(
                    f'{entry.logical_tensor}: direct graph output requires exactly one concrete kernel endpoint.'
                )
            if access_descriptor is not None:
                self.kernel_read_accesses.append(
                    {'endpoint': f'{producer_id}.{access_endpoints[0]}', 'descriptor': access_descriptor}
                )
            self.io_ports.append(
                {
                    'direction': 'output',
                    'port': int(graph_port),
                    'tensor': entry.logical_tensor,
                    'endpoint': endpoint,
                    'descriptor': descriptor,
                    'staging': staging,
                    'dtype': dtype,
                }
            )

    # ------------------------------------------------------------------
    # Memtile
    # ------------------------------------------------------------------

    def _emit_memtile(self, entry, p_ports, c_ports):
        unit = entry.unit
        if entry.producer.node:
            inst = self._kernel_inst(entry.producer.node)
            first_port = int(unit.producer_ports[0])
            base = inst.variant.describe_output_staging(
                entry.producer.node, inst.config, entry.producer.tensor, first_port, None
            )
            localize_descriptor(base, entry.producer.offset_base, entry.producer.buffer_dimension)
            shard_dim = int(unit.dimension)
            port_stride = int(unit.port_stride)
            unit_base_dim0 = int(unit.dimension_base)
            full_dims = list(base['buffer_dimension'])
            buf_dims = list(full_dims)
            buf_dims[shard_dim] = int(unit.dimension_size)
        else:
            if entry.graph_input is not None:
                base = graph_input_port_descriptor(entry, p_ports[0])
                buf_dims = list(unit.buffer_dimension)
            else:
                base = self._graph_input_writer_descriptor(entry)
                shard_dim = int(unit.dimension)
                port_stride = int(unit.port_stride)
                unit_base_dim0 = int(unit.dimension_base)
                full_dims = list(base['buffer_dimension'])
                buf_dims = list(full_dims)
                buf_dims[shard_dim] = int(unit.dimension_size)

        name = self._next_buffer_name(entry)
        buffer = {
            'name': name,
            'dimension': buf_dims,
            'num_buffers': 2,
            'ctype': self._buffer_ctype(entry),
            'writers': [],
            'readers': [],
            'tensor': entry.logical_tensor,
        }

        base_p = p_ports[0]
        for slot, p in enumerate(p_ports):
            if entry.producer.node is None:
                if entry.graph_input is not None:
                    desc = localized_graph_io_descriptor(
                        graph_input_writer_port_descriptor(entry, int(p)),
                        list(unit.offset_base),
                        list(buf_dims),
                    )
                else:
                    desc = self._graph_input_writer_descriptor(entry)
                    desc['buffer_dimension'] = list(buf_dims)
                    desc['offset'][shard_dim] = (int(p) - int(base_p)) * int(port_stride)
                self._max_graph_input_port = max(self._max_graph_input_port, int(p))
            else:
                inst = self._kernel_inst(entry.producer.node)
                desc = inst.variant.describe_output_staging(
                    entry.producer.node, inst.config, entry.producer.tensor, p, buf_dims
                )
                localize_descriptor(desc, entry.producer.offset_base, entry.producer.buffer_dimension)
                desc['buffer_dimension'] = list(buf_dims)
                desc['offset'][shard_dim] -= int(unit_base_dim0)

            source_type, source_endpoint = self._producer_endpoint_meta(entry.producer.node, entry.producer.group, p)
            buffer['writers'].append(
                {
                    'source': self._producer_endpoint(entry.producer.node, entry.producer.group, p),
                    'source_type': source_type,
                    'source_endpoint': source_endpoint,
                    'target': f'{name}.in[{slot}]',
                    'descriptor': desc,
                    'staging': self._graph_input_staging(entry, int(p)) if entry.producer.node is None else None,
                    'dtype': self._graph_input_dtype(entry).to_dict() if entry.producer.node is None else None,
                }
            )
            if entry.producer.node is None:
                self.io_ports.append(
                    {
                        'direction': 'input',
                        'port': int(p),
                        'tensor': entry.logical_tensor,
                        'endpoint': f'{name}.in[{slot}]',
                        'descriptor': desc,
                        'staging': buffer['writers'][-1]['staging'],
                        'dtype': buffer['writers'][-1]['dtype'],
                    }
                )

        if entry.consumers:
            consumer = entry.single_consumer()
            for local_out, i in enumerate(c_ports):
                if entry.producer.node is None and entry.graph_input is not None:
                    desc = localized_graph_io_descriptor(
                        graph_input_port_descriptor(entry, int(base_p + local_out)),
                        list(unit.offset_base),
                        list(buf_dims),
                    )
                else:
                    inst = self._kernel_inst(consumer.node)
                    desc = inst.variant.describe_input_staging(
                        consumer.node, inst.config, consumer.tensor, i, buf_dims, entry.producer.node
                    )
                    localize_descriptor(desc, consumer.offset_base, buf_dims)
                    desc['buffer_dimension'] = list(buf_dims)
                    desc['offset'][shard_dim] -= int(unit_base_dim0)
                    if int(unit.count) > 1:
                        desc['boundary_dimension'] = list(buf_dims)

                buffer['readers'].append(
                    {
                        'source': f'{name}.out[{local_out}]',
                        'target': (f'{sanitize_identifier(consumer.node.name)}.' f'{consumer.group}[{i}]'),
                        'target_type': 'op_impl',
                        'target_endpoint': {
                            'op_impl': consumer.node.name,
                            'op_impl_id': sanitize_identifier(consumer.node.name),
                            'group': consumer.group,
                            'port': int(i),
                        },
                        'descriptor': desc,
                    }
                )

        if entry.graph_output:
            reader_base = len(buffer['readers'])
            for slot, local_port in enumerate(p_ports):
                graph_port = self._next_graph_output_port
                self._next_graph_output_port += 1
                desc = self._graph_output_reader_descriptor(entry, local_port, buf_dims, unit_base_dim0=unit_base_dim0)
                buffer['readers'].append(
                    {
                        'source': f'{name}.out[{reader_base + slot}]',
                        'target': f'ofm[{graph_port}]',
                        'target_type': 'plio',
                        'target_endpoint': {'name': 'ofm', 'port': int(graph_port), 'op_impl_port': int(local_port)},
                        'descriptor': desc,
                        'staging': self._graph_output_staging(entry, int(local_port)),
                        'dtype': self._graph_output_dtype(entry).to_dict(),
                    }
                )
                self.io_ports.append(
                    {
                        'direction': 'output',
                        'port': int(graph_port),
                        'tensor': entry.logical_tensor,
                        'endpoint': f'{name}.out[{reader_base + slot}]',
                        'descriptor': desc,
                        'staging': buffer['readers'][-1]['staging'],
                        'dtype': buffer['readers'][-1]['dtype'],
                    }
                )

        self.buffers.append(buffer)

    # -------------------------------------------------------------------------
    # Graph IO descriptors
    # -------------------------------------------------------------------------

    def _graph_input_writer_descriptor(self, entry: EdgeEntry) -> Dict[str, Any]:
        consumer = entry.single_consumer()
        inst = self._kernel_inst(consumer.node)
        port = int(consumer.selected_ports(inst.ports.inputs[consumer.tensor].count)[0])
        base = inst.variant.describe_input_staging(consumer.node, inst.config, consumer.tensor, port, None, None)

        io_tile = list(base['io_tiling_dimension'])

        return {
            'access': 'write',
            'buffer_dimension': list(base['buffer_dimension']),
            'tiling_dimension': io_tile,
            'io_tiling_dimension': list(io_tile),
            'io_boundary_dimension': list(base['io_boundary_dimension']),
            'offset': [0 for _ in io_tile],
            'slice_dimension': int(base['slice_dimension']),
            'inner_dimension': int(base['inner_dimension']),
            'outer_dimension': int(base['outer_dimension']),
        }

    def _graph_output_reader_descriptor(
        self,
        entry: EdgeEntry,
        port: int,
        buf_dims: List[int],
        unit_base_dim0: int,
    ) -> Dict[str, Any]:
        producer = entry.producer
        inst = self._kernel_inst(producer.node)
        base = inst.variant.describe_output_staging(producer.node, inst.config, producer.tensor, port, buf_dims)
        shard_dim = int(base['slice_dimension'])
        io_tile = list(base['io_tiling_dimension'])
        io_boundary = list(base['io_boundary_dimension'])

        offset = list(base['offset'])
        offset[shard_dim] -= int(unit_base_dim0)
        boundary = list(io_boundary)
        boundary[shard_dim] = min(int(buf_dims[shard_dim]), max(0, int(io_boundary[shard_dim]) - int(unit_base_dim0)))
        return {
            'access': 'read',
            'buffer_dimension': list(buf_dims),
            'tiling_dimension': io_tile,
            'io_tiling_dimension': list(io_tile),
            'io_boundary_dimension': list(io_boundary),
            'offset': offset,
            'boundary_dimension': boundary,
            'slice_dimension': int(base['slice_dimension']),
            'inner_dimension': int(base['inner_dimension']),
            'outer_dimension': int(base['outer_dimension']),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _kernel_inst(self, node):
        return self.ctx.ir.execution.get(node.name) if node else None

    @staticmethod
    def _producer_endpoint(node, group, port):
        return f'ifm[{port}]' if node is None else f'{sanitize_identifier(node.name)}.{group}[{port}]'

    @staticmethod
    def _producer_endpoint_meta(node, group, port):
        if node is None:
            return 'plio', {'name': 'ifm', 'port': int(port)}
        return 'op_impl', {
            'op_impl': node.name,
            'op_impl_id': sanitize_identifier(node.name),
            'group': group,
            'port': int(port),
        }

    def _graph_input_role(self, entry: EdgeEntry) -> str:
        consumer = entry.single_consumer()
        role = input_role(consumer.node, consumer.tensor)
        if not role:
            raise RuntimeError(
                f'{entry.logical_tensor}: no role assigned on consumer {consumer.node.name!r}; '
                'frontend must call set_input_roles before building memory plan.'
            )
        return role

    def _buffer_ctype(self, entry):
        if entry.producer.node is None:
            dtype = self._graph_input_dtype(entry)
        else:
            dtype = self._graph_output_dtype(entry)
        return dtype.c_type

    def _graph_input_dtype(self, entry: EdgeEntry) -> AIEDataType:
        consumer = entry.single_consumer()
        inst = self._kernel_inst(consumer.node)
        role = self._graph_input_role(entry)
        return inst.variant.input_precision(inst.config, role)

    def _graph_input_staging(self, entry: EdgeEntry, port: int) -> Dict[str, Any]:
        if entry.graph_input is not None:
            return graph_input_writer_port_descriptor(entry, int(port))
        desc = self._graph_input_writer_descriptor(entry)
        shard_dim = int(entry.unit.dimension)
        port_stride = int(entry.unit.port_stride)
        tensor_local_port = int(port) - int(entry.unit.producer_tensor_port_base)
        desc['offset'][shard_dim] = int(tensor_local_port) * int(port_stride)
        return desc

    def _graph_output_dtype(self, entry: EdgeEntry) -> AIEDataType:
        producer = entry.producer
        inst = self._kernel_inst(producer.node)
        return inst.variant.output_precision(inst.config)

    def _graph_output_staging(self, entry: EdgeEntry, port: int) -> Dict[str, Any]:
        producer = entry.producer
        inst = self._kernel_inst(producer.node)
        base = inst.variant.describe_output_staging(producer.node, inst.config, producer.tensor, port, None)
        return _host_visible_output_staging(base)

    def _next_buffer_name(self, entry: EdgeEntry):
        base = sanitize_identifier(entry.producer.tensor)
        suffix = ''
        if int(entry.unit.count) > 1:
            suffix += f'_u{int(entry.unit.index)}'

        key = f'{base}{suffix}'
        idx = self._buffer_seq.get(key, 0) + 1
        self._buffer_seq[key] = idx

        stem = f'buffer_{base}{suffix}'
        return stem if idx == 1 else f'{stem}_{idx}'

    @staticmethod
    def _localize_direct_descriptor(descriptor: Dict[str, Any]) -> None:
        dimensions = staging_tile_shape(descriptor)
        offset = tuple(int(value) for value in descriptor['offset'])
        localize_descriptor(descriptor, offset, dimensions)


def _legalize_collected_entries(ctx, state):
    ctx.ir.physical.plan = {'_memory_plan_state': state}

    from ..placement import PlaceKernels
    from .classify import ClassifyTransportEntries
    from .fanout import LegalizeFanoutEntries
    from .memtile import LegalizeMemtilePortLimits

    LegalizeFanoutEntries().transform(ctx)
    ClassifyTransportEntries().transform(ctx)
    PlaceKernels().transform(ctx)
    LegalizeMemtilePortLimits().transform(ctx)
    return ctx.ir.physical.plan['_memory_plan_state']


def _host_visible_output_staging(base: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rewrite a producer output descriptor for host-visible graph output.

    Producer staging is expressed in padded kernel-buffer coordinates. Host-visible
    output uses IO boundary/tiling extents, so this helper swaps in
    io_boundary_dimension/io_tiling_dimension and rebases whole-slice offsets into
    IO-tile coordinates.
    """

    desc = dict(base)
    offsets = [int(x) for x in base['offset']]
    io_tile = [int(x) for x in base['io_tiling_dimension']]
    io_boundary = [int(x) for x in base['io_boundary_dimension']]
    traversal = list(base.get('tile_traversal', ()))

    host_offsets: List[int] = []
    for dim, offset in enumerate(offsets):
        if offset == 0:
            host_offsets.append(0)
            continue

        slice_extent = None
        for item in traversal:
            if int(item.get('dimension', -1)) != dim:
                continue
            stride = int(item.get('stride', 0))
            wrap = int(item.get('wrap', 0))
            if stride > 0 and wrap > 0:
                slice_extent = stride * wrap
                break

        if slice_extent and offset % slice_extent == 0:
            host_offsets.append((offset // slice_extent) * int(io_tile[dim]))
        else:
            host_offsets.append(offset)

    desc['buffer_dimension'] = list(io_boundary)
    desc['offset'] = host_offsets
    desc['tiling_dimension'] = list(io_tile)
    return desc


def _logical_stream_microtiled_2d_descriptor(base: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the AIE1 boundary reorder, or ``None`` when packed order is already linear."""

    dimensions = [int(value) for value in base['buffer_dimension']]
    tile = [int(value) for value in base['tiling_dimension']]
    if len(dimensions) != 2 or len(tile) != 2:
        raise NotImplementedError('AIE1 direct boundary DMA currently requires a 2-D kernel buffer.')

    inner = int(base['inner_dimension'])
    outer = int(base['outer_dimension'])
    micro_n = tile[inner]
    micro_m = tile[outer]
    if inner != 0 or outer != 1 or any(int(value) != 0 for value in base['offset']):
        raise NotImplementedError('AIE1 direct boundary DMA requires a local, inner-contiguous 2-D buffer.')
    if micro_m == 1:
        return None

    cols, rows = dimensions
    if cols % micro_n or rows % micro_m:
        raise ValueError(
            f'AIE1 direct boundary shape {(rows, cols)} is not divisible by microtile {(micro_m, micro_n)}.'
        )

    block = micro_m * micro_n
    desc = dict(base)
    # Physical storage is [row-block, column-block, row-in-block, column-in-block].
    # Splitting it into three DMA dimensions exposes logical rows without a tiler kernel.
    desc['buffer_dimension'] = [block, cols // micro_n, rows // micro_m]
    desc['tiling_dimension'] = [micro_n, 1, 1]
    desc['offset'] = [0, 0, 0]
    desc['tile_traversal'] = [
        {'dimension': 1, 'stride': 1, 'wrap': cols // micro_n},
        {'dimension': 0, 'stride': micro_n, 'wrap': micro_m},
        {'dimension': 2, 'stride': 1, 'wrap': rows // micro_m},
    ]
    desc.pop('boundary_dimension', None)
    return desc
