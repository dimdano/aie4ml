"""Insert the layout conversions ops ask for as kernel graphs of their own.

Resolution picks each op's kernel and says what layout it reads. Where an input arrives in another
one -- a strided conv's column-grouped frame, which neither the boundary nor a producing kernel
writes -- the op names a conversion, and this pass makes it an execution entry: a kernel with its own
ports, placement and staging, writing an execution-only value that the op then reads instead. Only
the execution graph changes; the logical graph keeps the model's semantics untouched.
"""

from __future__ import annotations

import logging

from ..ir import get_backend_context
from ..ir.graph import ExecutionEntry, ExecutionInput, ExecutionValue, OpNode, TensorContract
from ..op_impls.utils.io import normalized_staging
from .base import AIEPass

log = logging.getLogger(__name__)


def _insert(ctx, inst: ExecutionEntry, conversion) -> None:
    execution = ctx.ir.execution
    source = inst.input(conversion.source)
    variant, config = conversion.variant, conversion.config
    # The converter implements no logical op: its node only names it.
    node = OpNode(name=conversion.name, op_type=variant.op_type, dialect='aie')
    variant.validate_config(node, config, ctx.device)
    routes = inst.io_route.get('inputs', {})
    view = inst.port_views[conversion.source]
    entry = ExecutionEntry(
        node=node,
        variant=variant,
        ports=variant.build_ports(node, config),
        io_route={
            'inputs': {conversion.source: routes.get(conversion.source, 'auto')},
            'outputs': {conversion.target: 'direct'},
        },
        port_views={conversion.source: view, conversion.target: view},
        config=config,
        graph_header=variant.graph_header,
        graph_name=variant.graph_name,
        param_template=variant.param_template,
        inputs=(source,),
        outputs=(conversion.target,),
    )
    execution.insert_before(inst.name, entry)
    execution.add_value(ExecutionValue(conversion.target, producer=entry.name))
    execution.tensor_contracts[conversion.target] = TensorContract(
        contract=variant.output_staging_contract(node, config, conversion.target),
        port_staging=tuple(
            normalized_staging(variant.describe_output_staging(node, config, conversion.target, port, None))
            for port in range(int(variant.output_port_count(node, config)))
        ),
    )

    # The op now reads the converted value, straight from the converter.
    inst.inputs = tuple(
        ExecutionInput(conversion.target, item.role, shared_memory=conversion.shared_memory)
        if item.tensor == conversion.source
        else item
        for item in inst.inputs
    )
    inst.io_route = {
        **inst.io_route,
        'inputs': {**{k: v for k, v in routes.items() if k != conversion.source}, conversion.target: 'direct'},
    }
    inst.port_views = {**{k: v for k, v in inst.port_views.items() if k != conversion.source}, conversion.target: view}
    log.info('%s: reads %s through %s, a kernel on a tile of its own', inst.name, conversion.source, entry.name)


class LegalizeLayouts(AIEPass):
    def __init__(self):
        self.name = 'legalize_layouts'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        changed = False
        for inst in list(ctx.ir.execution):
            sources = {item.tensor: ctx.ir.execution.values[item.tensor] for item in inst.inputs}
            for conversion in inst.variant.input_conversions(inst.node, inst.config, sources):
                _insert(ctx, inst, conversion)
                changed = True
        ctx.ir.execution.verify()
        return changed
