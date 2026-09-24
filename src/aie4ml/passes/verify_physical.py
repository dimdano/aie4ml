"""The physical plan's invariants, checked before any code is generated.

Every kernel graph is placed, and every edge the execution graph requires in shared memory is realised
as one on each of its ports: a direct hand-over per port, legal by the rule placement searched under
(`shared_buffer`), its two ports pinned to the same memory, recorded as `shared_memory`, and carrying no
DMA access pattern.
"""

from __future__ import annotations

import re

from ..ir import get_backend_context
from .base import AIEPass
from .shared_buffer import SHARED_MEMORY, location_problem, pinned_locations, static_problem
from .utils import sanitize_identifier

_PORT = re.compile(r'^(?P<graph>\w+)\.(?P<group>\w+)\[(?P<port>\d+)\]$')


def _port(endpoint: str, graphs):
    match = _PORT.match(endpoint)
    if match is None or match['graph'] not in graphs:
        raise RuntimeError(f'physical plan endpoint {endpoint!r} names no placed kernel graph port.')
    return graphs[match['graph']], match['group'], int(match['port'])


def _pinned(ctx, inst, group: str, port: int):
    placement = ctx.ir.physical.placements[inst.name]
    return pinned_locations(inst, group, port, int(placement['col']), int(placement['row']))


def _kernel_ports(inst, group: str, port: int, direction: str):
    return {
        f'{sanitize_identifier(inst.name)}.{kernel_port}'
        for binding in getattr(inst.ports, direction).values()
        if binding.group == group
        for kernel_port in binding.endpoints[port]
    }


def verify_physical(ctx) -> None:
    execution, physical = ctx.ir.execution, ctx.ir.physical
    missing = [inst.name for inst in execution if inst.name not in physical.placements]
    if missing:
        raise RuntimeError(f'physical plan places no tile for {missing}.')
    graphs = {sanitize_identifier(inst.name): inst for inst in execution}
    accessed = {
        access['endpoint']
        for key in ('kernel_read_accesses', 'kernel_write_accesses')
        for access in physical.plan.get(key, ())
    }
    edges = physical.plan.get('direct_edges', ())
    for inst in execution:
        for item in inst.inputs:
            if not item.shared_memory:
                continue
            legs = [e for e in edges if e['tensor'] == item.tensor and _port(e['target'], graphs)[0] is inst]
            ports = inst.ports.inputs[item.tensor].count
            if len(legs) != ports:
                raise RuntimeError(
                    f'{inst.name}: must read {item.tensor!r} through shared memory on each of its {ports} ports, '
                    f'but the plan hands it over directly on {len(legs)}.'
                )
            for leg in legs:
                producer, p_group, p_port = _port(leg['source'], graphs)
                _, c_group, c_port = _port(leg['target'], graphs)
                problem = static_problem(ctx, item.tensor, producer, p_group, p_port, inst, c_group, c_port)
                problem = problem or location_problem(
                    _pinned(ctx, producer, p_group, p_port), _pinned(ctx, inst, c_group, c_port)
                )
                if not problem and leg.get('realization') != SHARED_MEMORY:
                    problem = f"the plan records it as {leg.get('realization')!r}"
                kernel_ports = _kernel_ports(producer, p_group, p_port, 'outputs') | _kernel_ports(
                    inst, c_group, c_port, 'inputs'
                )
                if not problem and accessed & kernel_ports:
                    problem = f'{sorted(accessed & kernel_ports)} carry a DMA access pattern'
                if problem:
                    raise RuntimeError(f'{leg["source"]} -> {leg["target"]}: must be shared memory, but {problem}.')


class VerifyPhysicalPlan(AIEPass):
    def __init__(self):
        self.name = 'verify_physical_plan'

    def transform(self, model_or_ctx) -> bool:
        verify_physical(get_backend_context(model_or_ctx))
        return False
