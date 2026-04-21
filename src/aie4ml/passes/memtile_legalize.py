from __future__ import annotations

import copy

from ..ir import get_backend_context
from .base import AIEPass
from .boundary_sharding import (
    graph_input_full_descriptor,
    graph_input_port_descs,
    graph_input_unit_box,
    graph_input_writer_port_descs,
)


class LegalizeMemtilePortLimits(AIEPass):
    def __init__(self):
        self.name = 'legalize_memtile_port_limits'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        state = ctx.ir.physical.plan['_memory_plan_state']
        entries = state['entries']

        max_in = int(ctx.device.max_mem_in_ports)
        max_out = int(ctx.device.max_mem_out_ports)
        rewritten = []
        changed = False
        next_graph_input_port = 0

        for entry in entries:
            if entry.topology_kind is None:
                raise RuntimeError(f'{entry.tensor}: missing transport topology; run transport classification first.')
            if entry.realization_kind is None:
                raise RuntimeError(
                    f'{entry.tensor}: missing transport realization; run transport classification first.'
                )
            if entry.topology_kind == 'relay':
                raise NotImplementedError(f'{entry.tensor}: transport topology relay is not implemented.')
            if entry.topology_kind == 'split':
                raise NotImplementedError(f'{entry.tensor}: transport topology split is not implemented.')
            if entry.topology_kind == 'join':
                raise NotImplementedError(f'{entry.tensor}: transport topology join is not implemented.')
            if len(entry.consumers) > 1:
                raise RuntimeError(
                    f'{entry.tensor}: unsupported join transport; memtile legalization requires '
                    f'post-fanout single-consumer entries, got {len(entry.consumers)} consumers.'
                )
            if entry.consumers and entry.graph_output:
                raise RuntimeError(
                    f'{entry.tensor}: unsupported boundary transport mix; '
                    'memtile legalization requires graph-output legs to be separated from consumer legs.'
                )

            p = max(1, int(entry.producer_ports))
            c = self._consumer_ports(entry, p, ctx)
            if entry.realization_kind == 'direct':
                if entry.producer is None or entry.graph_output:
                    raise RuntimeError(f'{entry.tensor}: direct realization is only valid for internal consumer edges.')
                if c != p:
                    raise RuntimeError(
                        f'{entry.tensor}: direct realization requires matching producer/consumer '
                        f'port counts ({p} != {c}).'
                    )
                legal = copy.copy(entry)
                legal.producer_ports = p
                legal.producer_port_ids = list(range(p))
                legal.producer_tensor_port_base = 0
                legal.consumer_port_ids = list(range(c))
                legal.shard_dim = None
                legal.shard_port_stride = None
                legal.shard_dim_base = 0
                legal.shard_dim_size = None
                legal.shard_offset_base = []
                legal.shard_buffer_dimension = []
                legal.graph_input_port_descs = {}
                legal.graph_input_writer_port_descs = {}
                legal.shard_index = 0
                legal.shard_count = 1
                rewritten.append(legal)
                continue

            units = max((p + max_in - 1) // max_in, (c + max_out - 1) // max_out)

            port_base = next_graph_input_port if entry.producer is None else 0
            if entry.producer is None:
                next_graph_input_port += p

            graph_input_descs = None if entry.producer is not None else graph_input_port_descs(entry, ctx, port_base)
            if entry.producer is None:
                shard_dim = port_stride = full_dim = None
            else:
                shard_dim, port_stride, full_dim = self._shard_params(entry, p, ctx)

            if units == 1:
                legal = copy.copy(entry)
                legal.producer_ports = p
                legal.producer_port_ids = [port_base + i for i in range(p)]
                legal.producer_tensor_port_base = int(port_base)
                legal.consumer_port_ids = list(range(c)) if entry.consumers else []
                if graph_input_descs is not None:
                    full = graph_input_full_descriptor(entry, ctx)
                    legal.shard_offset_base = [0 for _ in full['offset']]
                    legal.shard_buffer_dimension = list(full['buffer_dimension'])
                    legal.graph_input_port_descs = dict(graph_input_descs)
                    legal.graph_input_writer_port_descs = graph_input_writer_port_descs(graph_input_descs)
                else:
                    legal.shard_dim = int(shard_dim)
                    legal.shard_port_stride = int(port_stride)
                    legal.shard_dim_base = 0
                    legal.shard_dim_size = int(full_dim)
                legal.shard_index = 0
                legal.shard_count = 1
                self._validate_limits(legal, max_in, max_out)
                rewritten.append(legal)
                continue

            changed = True
            if entry.consumers:
                if c < p:
                    raise NotImplementedError(
                        f'{entry.tensor}: shard transport requires relay; '
                        f'one-stage sharding cannot contract producer_ports={p} to consumer_ports={c}.'
                    )
                if c % p != 0:
                    raise NotImplementedError(
                        f'{entry.tensor}: shard transport requires relay; '
                        f'one-stage sharding cannot regroup producer_ports={p} to consumer_ports={c}.'
                    )

            p_chunks = self._split_ports_serial(p, units)
            if entry.producer is None:
                p_chunks = [[port_base + i for i in chunk] for chunk in p_chunks]
            if entry.consumers:
                c_per_p = c // p
                if c_per_p * p != c:
                    raise RuntimeError(
                        f'{entry.tensor}: shard transport internal error; '
                        f'invalid consumer/producer port ratio (c={c}, p={p}).'
                    )
                if entry.producer is not None:
                    per_shard_out = max((len(p_ports) * c_per_p for p_ports in p_chunks), default=0)
                    if per_shard_out > max_out:
                        raise NotImplementedError(
                            f'{entry.tensor}: shard transport requires relay; '
                            f'one-stage sharding cannot realize producer_ports={p} -> consumer_ports={c} '
                            f'under memtile out-port limit {max_out}.'
                        )
                if entry.producer is not None:
                    c_chunks = self._consumer_chunks_for_internal_shard(entry, p, c, p_chunks, c_per_p, ctx)
                else:
                    c_chunks = []
                    start = 0
                    for p_ports in p_chunks:
                        size = len(p_ports) * c_per_p
                        c_chunks.append(list(range(start, start + size)))
                        start += size
                    if start != c:
                        raise RuntimeError(
                            f'{entry.tensor}: shard transport internal error; '
                            f'consumer chunking mismatch ({start} != {c}).'
                        )
            else:
                c_chunks = [[] for _ in range(units)]

            unit_boxes = None
            if graph_input_descs is None:
                unit_sizes = [len(chunk) * int(port_stride) for chunk in p_chunks]
                unit_bases = []
                acc = 0
                for sz in unit_sizes:
                    unit_bases.append(acc)
                    acc += int(sz)
            else:
                unit_boxes = [graph_input_unit_box(graph_input_descs, chunk) for chunk in p_chunks]

            for unit, (p_ports, c_ports) in enumerate(zip(p_chunks, c_chunks)):
                legal = copy.copy(entry)
                legal.producer_ports = len(p_ports)
                legal.producer_port_ids = list(p_ports)
                legal.producer_tensor_port_base = int(port_base)
                legal.consumer_port_ids = list(c_ports)
                if graph_input_descs is not None:
                    base, dims = unit_boxes[unit]
                    legal.shard_offset_base = list(base)
                    legal.shard_buffer_dimension = list(dims)
                    legal.graph_input_port_descs = dict(graph_input_descs)
                    legal.graph_input_writer_port_descs = graph_input_writer_port_descs(graph_input_descs)
                else:
                    legal.shard_dim = int(shard_dim)
                    legal.shard_port_stride = int(port_stride)
                    legal.shard_dim_base = int(unit_bases[unit])
                    legal.shard_dim_size = int(unit_sizes[unit])
                legal.shard_index = int(unit)
                legal.shard_count = int(units)
                self._validate_limits(legal, max_in, max_out)
                rewritten.append(legal)

        state['entries'] = rewritten
        return changed

    @staticmethod
    def _split_ports_serial(n: int, units: int):
        chunk = (n + units - 1) // units
        out = []
        start = 0
        for _ in range(units):
            size = min(chunk, n - start)
            out.append(list(range(start, start + size)))
            start += size
        return out

    @staticmethod
    def _consumer_chunks_for_internal_shard(
        entry, producer_ports: int, consumer_ports: int, p_chunks, c_per_p: int, ctx
    ):
        consumer = entry.consumers[0].consumer
        inst = ctx.ir.execution.get(consumer.name)
        shard_dim, port_stride, full_dim = LegalizeMemtilePortLimits._shard_params(entry, producer_ports, ctx)

        port_groups = {}
        for c_port in range(int(consumer_ports)):
            desc = inst.variant.describe_input_staging(
                consumer, inst.config, entry.tensor, c_port, None, entry.producer
            )
            offset = int(desc['offset'][int(shard_dim)])
            extent = int(desc['io_tiling_dimension'][int(shard_dim)])
            if offset < 0 or extent <= 0 or offset + extent > int(full_dim):
                raise NotImplementedError(
                    f'{entry.tensor}: shard transport requires relay; '
                    f'consumer port {c_port} range [{offset}, {offset + extent}) is invalid '
                    f'for producer shard dim{shard_dim} extent {full_dim}.'
                )
            group = offset // int(port_stride)
            group_limit = (group + 1) * int(port_stride)
            if group < 0 or group >= int(producer_ports) or offset + extent > group_limit:
                raise NotImplementedError(
                    f'{entry.tensor}: shard transport requires relay; '
                    f'consumer port {c_port} range [{offset}, {offset + extent}) crosses producer '
                    f'shard boundary for slice {group} on dim{shard_dim}.'
                )
            port_groups.setdefault(int(group), []).append(int(c_port))

        chunks = []
        for p_ports in p_chunks:
            chunk = []
            for p_port in p_ports:
                chunk.extend(port_groups.get(int(p_port), []))
            chunk = sorted(chunk)
            expected = len(p_ports) * int(c_per_p)
            if len(chunk) != expected:
                raise NotImplementedError(
                    f'{entry.tensor}: shard transport requires relay; '
                    f'expected {expected} consumer ports for producer slice set {p_ports}, got {len(chunk)}.'
                )
            chunks.append(chunk)

        assigned = sorted(port for chunk in chunks for port in chunk)
        if assigned != list(range(int(consumer_ports))):
            raise RuntimeError(
                f'{entry.tensor}: shard transport internal error; '
                f'consumer shard assignment mismatch ({assigned} != {list(range(int(consumer_ports)))}).'
            )
        return chunks

    @staticmethod
    def _consumer_ports(entry, producer_ports: int, ctx) -> int:
        if entry.consumers:
            consumer = entry.consumers[0].consumer
            inst = ctx.ir.execution.get(consumer.name)
            return int(inst.ports.inputs[entry.tensor].count)
        if entry.graph_output:
            return int(producer_ports)
        raise ValueError(f'{entry.tensor}: entry has neither consumers nor graph_output.')

    @staticmethod
    def _validate_limits(entry, max_in: int, max_out: int) -> None:
        in_ports = len(entry.producer_port_ids)
        if in_ports > max_in:
            raise RuntimeError(
                f'{entry.tensor}: shard transport internal error; '
                f'shard exceeds memtile in-port limit ({in_ports} > {max_in}).'
            )

        out_ports = (
            len(entry.consumer_port_ids)
            if entry.consumers
            else (len(entry.producer_port_ids) if entry.graph_output else 0)
        )
        if out_ports > max_out:
            raise RuntimeError(
                f'{entry.tensor}: shard transport internal error; '
                f'shard exceeds memtile out-port limit ({out_ports} > {max_out}).'
            )

    @staticmethod
    def _shard_params(entry, producer_ports: int, ctx):
        if entry.producer is not None:
            inst = ctx.ir.execution.get(entry.producer.name)
            d0 = inst.variant.describe_output_staging(entry.producer, inst.config, entry.tensor, 0, None)
            d1 = (
                inst.variant.describe_output_staging(entry.producer, inst.config, entry.tensor, 1, None)
                if producer_ports > 1
                else None
            )
        else:
            # TODO: graph-input stride is currently derived from consumer staging; if a repro
            # requires it, derive shard params directly from global graph-input shape/ports.
            consumer = entry.consumers[0].consumer
            inst = ctx.ir.execution.get(consumer.name)
            d0 = inst.variant.describe_input_staging(consumer, inst.config, entry.tensor, 0, None, None)
            d1 = (
                inst.variant.describe_input_staging(consumer, inst.config, entry.tensor, 1, None, None)
                if producer_ports > 1
                else None
            )

        shard_dim = int(d0['slice_dimension'])
        port_stride = (
            int(d1['offset'][shard_dim] - d0['offset'][shard_dim])
            if d1 is not None
            else int(d0['buffer_dimension'][shard_dim])
        )
        full_dim = int(d0['buffer_dimension'][shard_dim])

        if full_dim != port_stride * int(producer_ports):
            raise RuntimeError(
                f'{entry.tensor}: shard transport is not legal on dim{shard_dim}; '
                f'expected buffer_dimension[{shard_dim}] == port_stride * ports '
                f'({full_dim} != {port_stride} * {producer_ports}).'
            )

        return shard_dim, port_stride, full_dim
