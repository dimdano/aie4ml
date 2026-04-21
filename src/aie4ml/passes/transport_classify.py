from __future__ import annotations

from ..ir import get_backend_context
from .base import AIEPass
from .transport_legality import direct_transport_supported


class ClassifyTransportEntries(AIEPass):
    def __init__(self):
        self.name = 'classify_transport_entries'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        state = ctx.ir.physical.plan['_memory_plan_state']
        max_in = int(ctx.device.max_mem_in_ports)
        max_out = int(ctx.device.max_mem_out_ports)
        changed = False

        for entry in state['entries']:
            topology, staging_compatible = self._classify_entry(entry, ctx, max_in, max_out)
            realization = self._realization_kind(entry, topology, staging_compatible, ctx)
            if (
                entry.topology_kind != topology
                or entry.staging_compatible != staging_compatible
                or entry.realization_kind != realization
            ):
                changed = True
            entry.topology_kind = topology
            entry.staging_compatible = staging_compatible
            entry.realization_kind = realization

        return changed

    def _classify_entry(self, entry, ctx, max_in: int, max_out: int) -> tuple[str, bool | None]:
        if len(entry.consumers) > 1:
            raise RuntimeError(
                f'{entry.tensor}: unsupported join transport; '
                'classification requires post-fanout single-consumer entries.'
            )
        if entry.consumers and entry.graph_output:
            raise RuntimeError(
                f'{entry.tensor}: unsupported boundary transport mix; '
                'classification requires graph-output legs to be separated from consumer legs.'
            )
        if entry.producer is None or entry.graph_output:
            return 'boundary', None
        if not entry.consumers:
            raise RuntimeError(f'{entry.tensor}: transport classification requires a consumer or graph output.')

        producer_ports = max(1, int(entry.producer_ports))
        consumer_ports = self._consumer_ports(entry, producer_ports, ctx)
        consumer = entry.consumers[0].consumer

        # Direct AIE-to-AIE transport bypasses memtile entirely and is not
        # constrained by max_mem_in_ports / max_mem_out_ports.
        if self._has_consumer_perm(entry):
            return 'direct', False
        if producer_ports == consumer_ports:
            # Classification runs before memtile port-limit legalization, so the
            # default direct check uses canonical tensor-local ports 0..N-1. If a
            # caller has already populated explicit port IDs, preserve them.
            producer_port_ids = entry.producer_port_ids or list(range(producer_ports))
            consumer_port_ids = entry.consumer_port_ids or list(range(consumer_ports))
            if direct_transport_supported(
                ctx,
                entry.producer,
                consumer,
                entry.tensor,
                producer_port_ids,
                consumer_port_ids,
            ):
                return 'direct', True

        # Direct transport not feasible; classify using memtile port-limit constraints.
        units = max(
            (producer_ports + max_in - 1) // max_in,
            (consumer_ports + max_out - 1) // max_out,
        )
        if units > 1:
            if self._requires_relay(producer_ports, consumer_ports, units, max_out):
                entry.topology_kind = 'relay'
                entry.staging_compatible = None
                raise NotImplementedError(
                    f'{entry.tensor}: transport topology relay is not implemented; '
                    f'one-stage shard cannot realize producer_ports={producer_ports} '
                    f'-> consumer_ports={consumer_ports}.'
                )
            return 'shard', None
        return 'direct', False

    def _realization_kind(self, entry, topology: str, staging_compatible: bool | None, ctx) -> str:
        route = self._route_policy(entry, ctx)
        if route == 'direct':
            if topology != 'direct':
                raise RuntimeError(
                    f'{entry.tensor}: io_route=direct requested but transport topology is {topology}, not direct.'
                )
            if not staging_compatible:
                raise RuntimeError(
                    f'{entry.tensor}: io_route=direct requested but direct transport is not staging-compatible.'
                )
            return 'direct'
        if route == 'memtile':
            return 'memtile'
        if topology == 'direct' and staging_compatible:
            return 'direct'
        return 'memtile'

    @staticmethod
    def _route_policy(entry, ctx) -> str:
        if entry.producer is None or entry.graph_output:
            return 'memtile'

        modes = set()
        producer_inst = ctx.ir.execution.get(entry.producer.name)
        producer_mode = producer_inst.io_route.get('outputs', {}).get(entry.tensor)
        if producer_mode:
            modes.add(str(producer_mode))

        for consumer_conn in entry.consumers:
            consumer_inst = ctx.ir.execution.get(consumer_conn.consumer.name)
            consumer_mode = consumer_inst.io_route.get('inputs', {}).get(entry.tensor)
            if consumer_mode:
                modes.add(str(consumer_mode))

        bad = [mode for mode in modes if mode not in ('direct', 'memtile', 'auto')]
        if bad:
            raise ValueError(f'{entry.tensor}: unsupported io_route mode(s) {bad}.')
        if 'memtile' in modes:
            return 'memtile'
        if modes == {'direct'}:
            return 'direct'
        return 'auto'

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
    def _has_consumer_perm(entry) -> bool:
        consumer = entry.consumers[0].consumer
        trait = consumer.traits.get('io_view')
        if trait is None:
            return False
        inputs = trait.data.get('inputs', {})
        return inputs.get(entry.tensor, {}).get('perm') is not None

    @staticmethod
    def _requires_relay(producer_ports: int, consumer_ports: int, units: int, max_out: int) -> bool:
        if consumer_ports < producer_ports:
            return True
        if consumer_ports % producer_ports != 0:
            return True

        consumer_per_producer = consumer_ports // producer_ports
        max_ports_per_unit = (producer_ports + units - 1) // units
        return max_ports_per_unit * consumer_per_producer > max_out
