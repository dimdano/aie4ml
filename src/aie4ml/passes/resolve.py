from __future__ import annotations

from ..ir import get_backend_context
from ..ir.graph import TensorContract
from ..op_impls import get_family_resolver_registry
from ..op_impls.utils.io import ensure_io_view, normalized_staging, resolve_io_route
from .base import AIEPass


def _propagate_contracts(ctx, node, inst, config) -> None:
    """
    Propagates TensorContracts from producer outputs to consumer inputs based on the resolved execution entry.
    Requires LogicalIR.nodes to be in producer-before-consumer (topological) order.
    """
    for tensor in node.outputs:
        contract = inst.variant.output_staging_contract(node, config, tensor.name)
        if contract is None:
            continue
        port_count = inst.variant.output_port_count(node, config)
        ctx.ir.execution.tensor_contracts[tensor.name] = TensorContract(
            contract=contract,
            port_staging=tuple(
                normalized_staging(inst.variant.describe_output_staging(node, config, tensor.name, port, None))
                for port in range(int(port_count))
            ),
        )


def _resolved_input_contracts(ctx, node) -> dict[str, TensorContract]:
    return {
        tensor.name: ctx.ir.execution.tensor_contracts[tensor.name]
        for tensor in node.inputs
        if tensor.name in ctx.ir.execution.tensor_contracts
    }


def _same_execution_entry(inst, variant, ports, config) -> bool:
    return (
        inst.variant.variant_id == variant.variant_id
        and inst.ports == ports
        and inst.io_route == config.io_route
        and inst.io_views == config.io_views
        and inst.config == config
        and inst.graph_header == variant.graph_header
        and inst.graph_name == variant.graph_name
        and inst.param_template == variant.param_template
    )


class Resolve(AIEPass):
    """Resolve logical nodes into family-owned execution entries."""

    def __init__(self):
        self.name = 'resolve'
        self._registry = get_family_resolver_registry()

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        changed = False
        visited = set()

        ctx.ir.execution.tensor_contracts.clear()

        for node in ctx.ir.logical:
            if node.metadata['layer_class'] == 'Input':
                continue

            resolver = self._registry.get(node.op_type)
            ensure_io_view(node, ctx.device.generation)

            resolved_directives = dict(node.directives or {})
            resolved_directives['io_route'] = resolve_io_route(node)  # user intents
            resolved_directives['input_contracts'] = _resolved_input_contracts(ctx, node)

            config = resolver.resolve(node, ctx.device, resolved_directives)
            variant = resolver.select_variant(config, ctx.device.generation)
            variant.validate_config(node, config, ctx.device)
            ports = variant.build_ports(node, config)

            inst = ctx.ir.execution.get(node.name)
            if inst is not None and _same_execution_entry(inst, variant, ports, config):
                _propagate_contracts(ctx, node, inst, inst.config)
                visited.add(node.name)
                continue

            inst = ctx.ir.execution.register(
                node=node,
                variant=variant,
                ports=ports,
                io_route=dict(config.io_route),
                io_views=config.io_views,
                config=config,
                graph_header=variant.graph_header,
                graph_name=variant.graph_name,
                param_template=variant.param_template,
            )
            _propagate_contracts(ctx, node, inst, config)
            visited.add(node.name)
            changed = True

        if ctx.ir.execution.prune(visited):
            changed = True

        return changed
