from __future__ import annotations

import copy

from hls4ml.model.optimizer.optimizer import ModelOptimizerPass

from ..ir import get_backend_context


class LegalizeMemtilePortLimits(ModelOptimizerPass):
    def __init__(self):
        self.name = 'legalize_memtile_port_limits'

    def transform(self, model) -> bool:
        ctx = get_backend_context(model)
        state = ctx.ir.physical.plan['_memory_plan_state']
        entries = state['entries']

        max_in = int(ctx.device.max_mem_in_ports)
        max_out = int(ctx.device.max_mem_out_ports)
        rewritten = []
        changed = False

        for entry in entries:
            if len(entry.consumers) > 1:
                raise ValueError(
                    f'{entry.tensor}: LegalizeMemtilePortLimits received an entry with '
                    f'{len(entry.consumers)} consumers. LegalizeFanoutEntries must run first.'
                )
            if entry.consumers and entry.graph_output:
                raise ValueError(f'{entry.tensor}: mixed consumer + graph_output entry is not legal.')

            p = max(1, int(entry.producer_ports))
            c = self._consumer_ports(entry, p, ctx)
            units = max((p + max_in - 1) // max_in, (c + max_out - 1) // max_out)

            shard_dim, port_stride, full_dim = self._shard_params(entry, p, ctx)

            if units == 1:
                legal = copy.copy(entry)
                legal.producer_ports = p
                legal.producer_port_ids = list(range(p))
                legal.consumer_port_ids = list(range(c)) if entry.consumers else []
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
                    raise ValueError(
                        f'{entry.tensor}: units={units} requires consumer_ports >= producer_ports ' f'(p={p}, c={c}).'
                    )
                if c % p != 0:
                    raise ValueError(
                        f'{entry.tensor}: units={units} requires consumer_ports be a multiple of producer_ports '
                        f'(p={p}, c={c}).'
                    )

            p_chunks = self._split_ports_serial(p, units)
            if entry.consumers:
                c_per_p = c // p
                if c_per_p * p != c:
                    raise ValueError(f'{entry.tensor}: invalid consumer/producer port ratio (c={c}, p={p}).')
                c_chunks = []
                start = 0
                for p_ports in p_chunks:
                    size = len(p_ports) * c_per_p
                    c_chunks.append(list(range(start, start + size)))
                    start += size
                if start != c:
                    raise ValueError(f'{entry.tensor}: internal consumer chunking mismatch ({start} != {c}).')
            else:
                c_chunks = [[] for _ in range(units)]

            unit_sizes = [len(chunk) * int(port_stride) for chunk in p_chunks]
            unit_bases = []
            acc = 0
            for sz in unit_sizes:
                unit_bases.append(acc)
                acc += int(sz)

            for unit, (p_ports, c_ports) in enumerate(zip(p_chunks, c_chunks)):
                legal = copy.copy(entry)
                legal.producer_ports = len(p_ports)
                legal.producer_port_ids = list(p_ports)
                legal.consumer_port_ids = list(c_ports)
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
    def _consumer_ports(entry, producer_ports: int, ctx) -> int:
        if entry.consumers:
            consumer = entry.consumers[0].consumer
            inst = ctx.ir.execution.get(consumer.name)
            return int(inst.config.ports.inputs[entry.tensor].count)
        if entry.graph_output:
            return int(producer_ports)
        raise ValueError(f'{entry.tensor}: entry has neither consumers nor graph_output.')

    @staticmethod
    def _validate_limits(entry, max_in: int, max_out: int) -> None:
        in_ports = len(entry.producer_port_ids)
        if in_ports > max_in:
            raise ValueError(f'{entry.tensor}: shard exceeds memtile in-port limit ({in_ports} > {max_in}).')

        out_ports = (
            len(entry.consumer_port_ids)
            if entry.consumers
            else (len(entry.producer_port_ids) if entry.graph_output else 0)
        )
        if out_ports > max_out:
            raise ValueError(f'{entry.tensor}: shard exceeds memtile out-port limit ({out_ports} > {max_out}).')

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
            raise ValueError(
                f'{entry.tensor}: cannot shard dim{shard_dim}; '
                f'expected buffer_dimension[{shard_dim}] == port_stride * ports '
                f'({full_dim} != {port_stride} * {producer_ports}).'
            )

        return shard_dim, port_stride, full_dim
