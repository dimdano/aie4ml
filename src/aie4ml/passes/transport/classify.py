from __future__ import annotations

from ...ir import get_backend_context
from ..base import AIEPass
from .legality import direct_transport_failure, memtile_staging_failure, uses_stream
from .model import TransportDecision


class ClassifyTransportEntries(AIEPass):
    """Choose direct or memtile realization without performing memtile legalization."""

    def __init__(self):
        self.name = 'classify_transport_entries'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        entries = ctx.ir.physical.plan['_memory_plan_state']['entries']
        changed = False
        for entry in entries:
            decision = self._classify_entry(entry, ctx)
            changed = changed or entry.decision != decision
            entry.decision = decision
        return changed

    def _classify_entry(self, entry, ctx) -> TransportDecision:
        self._validate_entry(entry)
        route = self._route_policy(entry, ctx)
        is_boundary = entry.producer.node is None or entry.graph_output
        has_memtile = bool(ctx.device.has_memtile)

        restage_failure = memtile_staging_failure(ctx, entry)
        if route == 'memtile' and restage_failure is not None:
            raise NotImplementedError(
                f'{entry.logical_tensor}: io_route=memtile needs re-staging, but {restage_failure}.'
            )
        can_memtile = has_memtile and restage_failure is None

        if uses_stream(ctx, entry):
            # A stream port has no buffer for a memory tile to stage, so the leg is
            # point-to-point (or PLIO) and must already agree on its element order.
            if route == 'memtile':
                raise RuntimeError(f'{entry.logical_tensor}: io_route=memtile requested on a stream port.')
            if not is_boundary:
                consumer = entry.single_consumer()
                if self._has_consumer_perm(consumer):
                    failure = f'consumer {consumer.node.name}.{consumer.group} applies an input permutation'
                else:
                    failure = direct_transport_failure(ctx, entry.logical_tensor, entry.producer, consumer)
                if failure is not None:
                    raise RuntimeError(
                        f'{entry.logical_tensor}: point-to-point ports cannot connect directly: {failure}.'
                    )
            return TransportDecision('direct', True)

        if is_boundary:
            if route == 'memtile' and not has_memtile:
                raise RuntimeError(
                    f'{entry.logical_tensor}: io_route=memtile requested on a device without memory tiles.'
                )
            realization = 'direct' if route == 'direct' or (route == 'auto' and not can_memtile) else 'memtile'
            return TransportDecision(realization, True if realization == 'direct' else None)

        consumer = entry.single_consumer()
        if self._has_consumer_perm(consumer):
            direct_failure = f'consumer {consumer.node.name}.{consumer.group} applies an input permutation'
        else:
            direct_failure = direct_transport_failure(ctx, entry.logical_tensor, entry.producer, consumer)
        staging_compatible = direct_failure is None
        if route == 'direct':
            if not staging_compatible:
                raise RuntimeError(
                    f'{entry.logical_tensor}: io_route=direct requested but point-to-point transport '
                    f'is not staging-compatible: {direct_failure}.'
                )
            realization = 'direct'
        elif route == 'memtile':
            if not has_memtile:
                raise RuntimeError(
                    f'{entry.logical_tensor}: io_route=memtile requested on a device without memory tiles.'
                )
            realization = 'memtile'
        else:
            if staging_compatible:
                realization = 'direct'
            elif not can_memtile:
                raise RuntimeError(
                    f'{entry.logical_tensor}: AIE1 cannot directly connect this transport: {direct_failure}; '
                    'relay/relayout is not implemented.'
                )
            else:
                realization = 'memtile'
        return TransportDecision(realization, staging_compatible)

    @staticmethod
    def _validate_entry(entry) -> None:
        if len(entry.consumers) > 1:
            raise RuntimeError(
                f'{entry.logical_tensor}: classification requires fanout to be split into single-consumer entries.'
            )
        if entry.consumers and entry.graph_output:
            raise RuntimeError(
                f'{entry.logical_tensor}: classification requires graph-output and consumer legs to be separate.'
            )
        if entry.producer.node is not None and not entry.graph_output and not entry.consumers:
            raise RuntimeError(f'{entry.logical_tensor}: internal transport entry has no consumer.')

    @staticmethod
    def _route_policy(entry, ctx) -> str:
        modes = set()
        if entry.producer.node is not None:
            producer_inst = ctx.ir.execution.get(entry.producer.node.name)
            producer_mode = producer_inst.io_route.get('outputs', {}).get(entry.producer.tensor)
            if producer_mode:
                modes.add(str(producer_mode))

        if entry.consumers:
            consumer = entry.single_consumer()
            consumer_inst = ctx.ir.execution.get(consumer.node.name)
            consumer_mode = consumer_inst.io_route.get('inputs', {}).get(consumer.tensor)
            if consumer_mode:
                modes.add(str(consumer_mode))

        bad = [mode for mode in modes if mode not in ('direct', 'memtile', 'auto')]
        if bad:
            raise ValueError(f'{entry.logical_tensor}: unsupported io_route mode(s) {bad}.')
        if 'memtile' in modes:
            return 'memtile'
        if modes == {'direct'}:
            return 'direct'
        return 'auto'

    @staticmethod
    def _has_consumer_perm(consumer) -> bool:
        trait = consumer.node.traits.get('io_view')
        if trait is None:
            return False
        return trait.data.get('inputs', {}).get(consumer.tensor, {}).get('perm') is not None
