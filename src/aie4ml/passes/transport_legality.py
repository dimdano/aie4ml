from __future__ import annotations

from typing import Sequence

from ..op_impls.utils.io import normalized_staging


def direct_transport_supported(
    ctx, producer, consumer, tensor_name: str, producer_ports: Sequence[int], consumer_ports: Sequence[int]
) -> bool:
    if producer is None or consumer is None or len(producer_ports) != len(consumer_ports):
        return False

    src_inst = ctx.ir.execution.get(producer.name)
    dst_inst = ctx.ir.execution.get(consumer.name)
    if src_inst is None or dst_inst is None:
        raise RuntimeError(f'{tensor_name}: direct transport legality requires resolved execution instances.')

    tc = ctx.ir.execution.tensor_contracts.get(tensor_name)
    if tc is not None:
        if len(producer_ports) != len(tc.port_staging) or len(consumer_ports) != len(tc.port_staging):
            return False
        if src_inst.io_views.get(tensor_name) is None or dst_inst.io_views.get(tensor_name) is None:
            return False

    for p_port, c_port in zip(producer_ports, consumer_ports):
        src_desc = src_inst.variant.describe_output_staging(producer, src_inst.config, tensor_name, int(p_port), None)
        dst_desc = dst_inst.variant.describe_input_staging(
            consumer,
            dst_inst.config,
            tensor_name,
            int(c_port),
            None,
            producer,
        )
        if normalized_staging(src_desc) != normalized_staging(dst_desc):
            return False
    return True
