from __future__ import annotations

import copy

from ...op_impls.utils.io import normalized_staging
from .descriptors import localize_descriptor
from .model import Endpoint


def direct_transport_failure(
    ctx,
    logical_tensor: str,
    producer: Endpoint,
    consumer: Endpoint,
) -> str | None:
    if producer.node is None or consumer.node is None:
        return 'direct transport requires resolved kernel endpoints'
    producer_inst = ctx.ir.execution.get(producer.node.name)
    consumer_inst = ctx.ir.execution.get(consumer.node.name)
    if producer_inst is None or consumer_inst is None:
        raise RuntimeError(f'{logical_tensor}: direct transport legality requires resolved execution instances.')

    producer_ports = producer.selected_ports(producer_inst.ports.outputs[producer.tensor].count)
    consumer_ports = consumer.selected_ports(consumer_inst.ports.inputs[consumer.tensor].count)
    if len(producer_ports) != len(consumer_ports):
        return f'producer ports {producer_ports} do not match consumer ports {consumer_ports}'

    tc = ctx.ir.execution.tensor_contracts.get(producer.tensor)
    if tc is not None:
        if any(int(port) < 0 or int(port) >= len(tc.port_staging) for port in producer_ports):
            return f'producer ports {producer_ports} exceed the published staging contract'
        if producer_inst.io_views.get(producer.tensor) is None:
            return f'producer tensor {producer.tensor!r} has no resolved I/O view'
        if consumer_inst.io_views.get(consumer.tensor) is None:
            return f'consumer tensor {consumer.tensor!r} has no resolved I/O view'

    for p_port, c_port in zip(producer_ports, consumer_ports):
        src_desc = producer_inst.variant.describe_output_staging(
            producer.node, producer_inst.config, producer.tensor, int(p_port), None
        )
        if producer.offset_base:
            src_desc = copy.deepcopy(src_desc)
            localize_descriptor(src_desc, producer.offset_base, producer.buffer_dimension)
        dst_desc = consumer_inst.variant.describe_input_staging(
            consumer.node,
            consumer_inst.config,
            consumer.tensor,
            int(c_port),
            None,
            producer.node,
        )
        dst_desc = copy.deepcopy(dst_desc)
        localize_descriptor(dst_desc, consumer.offset_base, src_desc['buffer_dimension'])
        src_staging = normalized_staging(src_desc)
        dst_staging = normalized_staging(dst_desc)
        if src_staging != dst_staging:
            keys = sorted(
                key for key in set(src_staging) | set(dst_staging) if src_staging.get(key) != dst_staging.get(key)
            )
            return (
                f'staging mismatch at {producer.node.name}.{producer.group}[{p_port}] -> '
                f'{consumer.node.name}.{consumer.group}[{c_port}] ({", ".join(keys)})'
            )
    return None
