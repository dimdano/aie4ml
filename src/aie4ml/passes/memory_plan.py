# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hls4ml.model.optimizer.optimizer import ModelOptimizerPass

from ..ir import OpNode, get_backend_context
from .utils import sanitize_identifier


@dataclass
class _Connection:
    producer: Optional[OpNode]
    consumer: Optional[OpNode]
    producer_group: str
    consumer_group: str
    tensor: str
    external_kind: Optional[str] = None


@dataclass
class _EdgeEntry:
    tensor: str
    producer: Optional[OpNode]
    producer_group: str
    producer_ports: int
    consumers: List[_Connection] = field(default_factory=list)
    graph_output: bool = False


class BuildMemoryPlan(ModelOptimizerPass):
    def __init__(self):
        self.name = 'plan_memory'

    def transform(self, model) -> bool:
        ctx = get_backend_context(model)
        ctx.ir.physical.plan = _CodegenPlanner(ctx).build(list(ctx.ir.logical))
        return True


class _CodegenPlanner:
    def __init__(self, ctx):
        self.ctx = ctx
        self.device = ctx.device

        self.buffers = []
        self.direct_edges = []
        self.layer_indices = {}

    def build(self, nodes):
        idx = 0
        for n in nodes:
            if self._kernel_inst(n):
                idx += 1
                self.layer_indices[n.name] = idx

        conns = self._collect_connections(nodes)
        entries = self._group_edges(conns)

        for e in entries:
            self._materialize_entry(e)

        return {
            'buffers': self.buffers,
            'direct_edges': self.direct_edges,
        }

    # -------------------------------------------------------------------------
    # Connections
    # -------------------------------------------------------------------------

    def _collect_connections(self, nodes: List[OpNode]) -> List[_Connection]:
        producers: Dict[str, Tuple[OpNode, str]] = {}

        # collect producers
        for n in nodes:
            inst = self._kernel_inst(n)
            if not inst:
                continue
            for t in getattr(n, 'outputs', []):
                tname = t.name
                pg = inst.config['parameters']['ports']['outputs'][tname]['group']
                producers[tname] = (n, pg)

        connections: List[_Connection] = []
        seen_outputs: set[str] = set()

        # inputs
        for n in nodes:
            inst = self._kernel_inst(n)
            if not inst:
                continue
            for t in getattr(n, 'inputs', []):
                tname = t.name
                cg = inst.config['parameters']['ports']['inputs'][tname]['group']
                if tname in producers:
                    p, pg = producers[tname]
                    connections.append(_Connection(p, n, pg, cg, tname))
                    seen_outputs.add(tname)
                else:
                    connections.append(_Connection(None, n, 'graph_input', cg, tname, 'input'))

        # graph outputs
        for n in nodes:
            inst = self._kernel_inst(n)
            if not inst:
                continue
            for t in getattr(n, 'outputs', []):
                tname = t.name
                if tname not in seen_outputs:
                    pg = inst.config['parameters']['ports']['outputs'][tname]['group']
                    connections.append(_Connection(n, None, pg, 'graph_output', tname, 'output'))

        return connections

    # -------------------------------------------------------------------------
    # Group edges
    # -------------------------------------------------------------------------

    def _group_edges(self, connections: Iterable[_Connection]) -> List[_EdgeEntry]:
        grouped: Dict[Tuple[str, str], _EdgeEntry] = {}

        for c in connections:
            key = (c.tensor, c.producer_group)
            if key not in grouped:
                grouped[key] = _EdgeEntry(
                    tensor=c.tensor,
                    producer=c.producer,
                    producer_group=c.producer_group,
                    producer_ports=self._producer_port_count(c.producer, c.tensor),
                )

            e = grouped[key]

            if c.external_kind == 'output':
                e.graph_output = True
            else:
                e.consumers.append(c)
                if c.external_kind == 'input':
                    e.producer_ports = max(
                        e.producer_ports,
                        self._consumer_port_count(c.consumer, c.tensor),
                    )

        return list(grouped.values())

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def _materialize_entry(self, entry: _EdgeEntry):
        p = max(1, entry.producer_ports)
        c = sum(self._consumer_port_count(x.consumer, entry.tensor) for x in entry.consumers)

        units = self._required_units(p, c)

        # DIRECT ONLY IF units == 1 and strictly single-port 1:1
        if (
            units == 1
            and entry.producer
            and not entry.graph_output
            and len(entry.consumers) == 1
            and p == 1
            and self._consumer_port_count(entry.consumers[0].consumer, entry.tensor) == 1
        ):
            self._emit_direct(entry, p)
        else:
            self._emit_memtile(entry, p, units)

    # ------------------------------------------------------------------
    # Direct
    # ------------------------------------------------------------------

    def _emit_direct(self, entry, ports):
        p = entry.producer
        c = entry.consumers[0]

        for i in range(ports):
            self.direct_edges.append(
                {
                    'source': f'{sanitize_identifier(p.name)}.{entry.producer_group}[{i}]',
                    'target': f'{sanitize_identifier(c.consumer.name)}.{c.consumer_group}[{i}]',
                    'tensor': entry.tensor,
                }
            )

    # ------------------------------------------------------------------
    # Memtile
    # ------------------------------------------------------------------

    def _emit_memtile(self, entry, producer_ports, units):
        consumers = entry.consumers
        consumer_ports = sum(self._consumer_port_count(c.consumer, entry.tensor) for c in consumers)
        if entry.graph_output:
            consumer_ports = producer_ports
        if units > 1:
            if consumer_ports < producer_ports:
                raise RuntimeError(
                    f'{entry.tensor}: units={units} requires consumer_ports >= producer_ports '
                    f'(p={producer_ports}, c={consumer_ports}).'
                )
            if consumer_ports % producer_ports != 0:
                raise RuntimeError(
                    f'{entry.tensor}: units={units} requires consumer_ports be a multiple of producer_ports '
                    f'(p={producer_ports}, c={consumer_ports}).'
                )

        # SERIAL port split (contiguous blocks; unit0 gets first chunk, unit1 next, ...)
        p_chunks = self._split_ports_serial(producer_ports, units)
        if units == 1:
            c_chunks = [list(range(consumer_ports))]
        else:
            mult = consumer_ports // producer_ports
            c_chunks = []
            start = 0
            for p_ports in p_chunks:
                size = len(p_ports) * mult
                c_chunks.append(list(range(start, start + size)))
                start += size
            if start != consumer_ports:
                raise RuntimeError(
                    f'{entry.tensor}: internal error computing consumer sharding '
                    f'(expected {consumer_ports} ports, assigned {start}).'
                )

        # Determine shard stride per port in buffer space (required for sharding/rebasing).
        if entry.producer:
            inst = self._kernel_inst(entry.producer)
            d0 = inst.variant.describe_output_staging(entry.producer, inst.attributes, entry.tensor, 0, None, None)
            d1 = (
                inst.variant.describe_output_staging(entry.producer, inst.attributes, entry.tensor, 1, None, None)
                if producer_ports > 1
                else None
            )
            shard_dim = int(d0['slice_dimension'])
            port_stride0 = (
                int(d1['offset'][shard_dim] - d0['offset'][shard_dim])
                if d1 is not None
                else int(d0['buffer_dimension'][shard_dim])
            )
            full_dim0 = int(d0['buffer_dimension'][shard_dim])
        else:
            c0 = consumers[0].consumer
            inst = self._kernel_inst(c0)
            d0 = inst.variant.describe_input_staging(c0, inst.attributes, entry.tensor, 0, None, None, None)
            d1 = (
                inst.variant.describe_input_staging(c0, inst.attributes, entry.tensor, 1, None, None, None)
                if producer_ports > 1
                else None
            )
            shard_dim = int(d0['slice_dimension'])
            port_stride0 = (
                int(d1['offset'][shard_dim] - d0['offset'][shard_dim])
                if d1 is not None
                else int(d0['buffer_dimension'][shard_dim])
            )
            full_dim0 = int(d0['buffer_dimension'][shard_dim])
        if full_dim0 != port_stride0 * int(producer_ports):
            raise RuntimeError(
                f'{entry.tensor}: cannot shard dim{shard_dim}; '
                f'expected buffer_dimension[{shard_dim}] == port_stride * ports '
                f'({full_dim0} != {port_stride0} * {producer_ports}).'
            )

        # Prefix sums of sharded dim0 sizes, used for offset rebasing.
        unit_dim0_sizes = [len(chunk) * port_stride0 for chunk in p_chunks]
        unit_dim0_bases: List[int] = []
        acc = 0
        for size in unit_dim0_sizes:
            unit_dim0_bases.append(acc)
            acc += int(size)

        for u in range(units):
            p_ports = p_chunks[u]
            c_ports = c_chunks[u]
            if not p_ports and not c_ports:
                continue
            if len(p_ports) > self.device.max_mem_in_ports:
                raise RuntimeError(
                    f'{entry.tensor}: unit {u+1} exceeds memtile in-port limit '
                    f'({len(p_ports)} > {self.device.max_mem_in_ports}).'
                )
            if len(c_ports) > self.device.max_mem_out_ports:
                raise RuntimeError(
                    f'{entry.tensor}: unit {u+1} exceeds memtile out-port limit '
                    f'({len(c_ports)} > {self.device.max_mem_out_ports}).'
                )

            # base descriptor
            if entry.producer:
                inst = self._kernel_inst(entry.producer)
                base = inst.variant.describe_output_staging(
                    entry.producer, inst.attributes, entry.tensor, 0, None, None
                )
            else:
                base = self._graph_input_writer_descriptor(entry)

            full_dims = list(base['buffer_dimension'])
            shard_size = int(unit_dim0_sizes[u])
            buf_dims = list(full_dims)
            buf_dims[shard_dim] = shard_size
            unit_base_dim0 = int(unit_dim0_bases[u])

            name = self._buffer_name(entry.tensor, u, units)

            buffer = {
                'name': name,
                'dimension': buf_dims,
                'num_buffers': 2,
                'ctype': self._buffer_ctype(entry),
                'writers': [],
                'readers': [],
                'tensor': entry.tensor,
            }

            # writers
            base_p = p_ports[0] if p_ports else 0
            for slot, p in enumerate(p_ports):
                if entry.producer is None:
                    desc = self._graph_input_writer_descriptor(entry)
                    desc['buffer_dimension'] = list(buf_dims)
                    desc['offset'][shard_dim] = (p - base_p) * port_stride0
                else:
                    inst = self._kernel_inst(entry.producer)
                    staging = self._output_staging(inst, entry.tensor)
                    desc = inst.variant.describe_output_staging(
                        entry.producer, inst.attributes, entry.tensor, p, buf_dims, staging
                    )
                    desc['buffer_dimension'] = list(buf_dims)
                    desc['offset'][shard_dim] -= unit_base_dim0

                buffer['writers'].append(
                    {
                        'source': self._producer_endpoint(entry.producer, entry.producer_group, p),
                        'source_type': self._producer_endpoint_meta(entry.producer, entry.producer_group, p)[0],
                        'source_endpoint': self._producer_endpoint_meta(entry.producer, entry.producer_group, p)[1],
                        'target': f'{name}.in[{slot}]',
                        'descriptor': desc,
                    }
                )

            # readers
            base_c = c_ports[0] if c_ports else 0
            flat = 0
            for c in consumers:
                for i in range(self._consumer_port_count(c.consumer, entry.tensor)):
                    if flat not in c_ports:
                        flat += 1
                        continue

                    inst = self._kernel_inst(c.consumer)
                    staging = self._input_staging(inst, entry.tensor)
                    desc = inst.variant.describe_input_staging(
                        c.consumer, inst.attributes, entry.tensor, i, buf_dims, staging, entry.producer
                    )
                    desc['buffer_dimension'] = list(buf_dims)
                    desc['offset'][shard_dim] -= unit_base_dim0
                    if units > 1:
                        desc['boundary_dimension'] = list(buf_dims)
                    local_out = flat - base_c

                    buffer['readers'].append(
                        {
                            'source': f'{name}.out[{local_out}]',
                            'target': f'{sanitize_identifier(c.consumer.name)}.{c.consumer_group}[{i}]',
                            'target_type': 'kernel',
                            'target_endpoint': {
                                'kernel': c.consumer.name,
                                'kernel_id': sanitize_identifier(c.consumer.name),
                                'group': c.consumer_group,
                                'port': int(i),
                            },
                            'descriptor': desc,
                        }
                    )
                    flat += 1

            if entry.graph_output:
                for slot, port in enumerate(p_ports):
                    desc = self._graph_output_reader_descriptor(
                        entry,
                        port,
                        buf_dims,
                        unit_base_dim0=unit_base_dim0,
                    )
                    buffer['readers'].append(
                        {
                            'source': f'{name}.out[{slot}]',
                            'target': f'ofm[{port}]',
                            'target_type': 'plio',
                            'target_endpoint': {'name': 'ofm', 'port': int(port)},
                            'descriptor': desc,
                        }
                    )

            self.buffers.append(buffer)

    # -------------------------------------------------------------------------
    # Graph IO descriptors
    # -------------------------------------------------------------------------

    def _graph_input_writer_descriptor(self, entry: _EdgeEntry) -> Dict[str, Any]:
        c = entry.consumers[0].consumer
        inst = self._kernel_inst(c)
        base = inst.variant.describe_input_staging(c, inst.attributes, entry.tensor, 0, None, None, None)

        io_tile = list(base['io_tiling_dimension'])

        return {
            'access': 'write',
            'buffer_dimension': list(base['buffer_dimension']),
            'tiling_dimension': io_tile,
            'offset': [0 for _ in io_tile],
        }

    def _graph_output_reader_descriptor(
        self,
        entry: _EdgeEntry,
        port: int,
        buf_dims: List[int],
        unit_base_dim0: int,
    ) -> Dict[str, Any]:
        p = entry.producer
        inst = self._kernel_inst(p)
        base = inst.variant.describe_output_staging(p, inst.attributes, entry.tensor, port, buf_dims, None)
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
            'offset': offset,
            'boundary_dimension': boundary,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _input_staging(self, inst, tensor):
        st = inst.attributes.staging['inputs']
        return st[tensor] if tensor in st else None

    def _output_staging(self, inst, tensor):
        st = inst.attributes.staging['outputs']
        return st[tensor] if tensor in st else None

    def _split_ports_serial(self, n, units):
        chunk = (n + units - 1) // units
        out = []
        start = 0
        for _ in range(units):
            size = min(chunk, n - start)
            out.append(list(range(start, start + size)))
            start += size
        return out

    def _kernel_inst(self, node):
        return self.ctx.ir.kernels.get(node.name) if node else None

    def _producer_port_count(self, node, tensor):
        if node is None:
            return 1
        return self._kernel_inst(node).config['parameters']['ports']['outputs'][tensor]['count']

    def _consumer_port_count(self, node, tensor):
        return self._kernel_inst(node).config['parameters']['ports']['inputs'][tensor]['count']

    def _producer_endpoint(self, node, group, port):
        return f'ifm[{port}]' if node is None else f'{sanitize_identifier(node.name)}.{group}[{port}]'

    def _producer_endpoint_meta(self, node, group, port):
        if node is None:
            return 'plio', {'name': 'ifm', 'port': int(port)}
        return 'kernel', {
            'kernel': node.name,
            'kernel_id': sanitize_identifier(node.name),
            'group': group,
            'port': int(port),
        }

    def _consumer_endpoint_meta(self, node, group, port):
        if node is None:
            return 'plio', {'name': 'ofm', 'port': int(port)}
        return 'kernel', {
            'kernel': node.name,
            'kernel_id': sanitize_identifier(node.name),
            'group': group,
            'port': int(port),
        }

    def _buffer_ctype(self, entry):
        if entry.producer is None:
            c = entry.consumers[0].consumer
            return f'typename Cfg{self.layer_indices[c.name]}::data_t'
        return f'typename Cfg{self.layer_indices[entry.producer.name]}::result_t'

    def _required_units(self, p, c):
        return max(
            (p + self.device.max_mem_in_ports - 1) // self.device.max_mem_in_ports,
            (c + self.device.max_mem_out_ports - 1) // self.device.max_mem_out_ports,
        )

    @staticmethod
    def _buffer_name(tensor, idx, total):
        base = sanitize_identifier(tensor)
        return f'buffer_{base}' if total == 1 else f'buffer_{base}_u{idx+1}'
