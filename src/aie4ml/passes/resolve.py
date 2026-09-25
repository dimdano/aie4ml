from __future__ import annotations

from ..ir import get_backend_context
from ..ir.graph import ExecutionInput, ExecutionValue, ExecutionView, TensorContract, ViewPart, input_role
from ..op_impls import get_family_resolver_registry
from ..op_impls.utils.io import check_io_view, normalized_staging, resolve_io_route
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
    return inst.variant is variant and inst.ports == ports and inst.config == config


def _check_transposed_views(node, config, variant) -> None:
    """A folded transpose needs both halves: the DMA walks the microtile grid in view order and
    the kernel transposes each block on load. Refuse rather than feed a kernel permuted data.
    """

    for name, view in (getattr(config, 'io_views', None) or {}).items():
        if not view.is_transposed:
            continue
        if not variant.kernel_transposes_microtile:
            raise NotImplementedError(
                f'{node.name}: {name!r} is a transposed view, but {variant.variant_id} does not '
                'transpose the microtile on load, so the kernel would read permuted data.'
            )
        if view.microtile is None:
            raise NotImplementedError(
                f'{node.name}: {name!r} is a transposed view staged in whole rows; the DMA needs a '
                'microtiled staging to walk the grid in view order.'
            )


def _folded_views(node):
    """(value, view) for each output of a folded slice, split or concat, checked against the trait's schema."""

    def trait(name, keys, part_keys):
        data = node.traits[name].data
        if set(data) != keys or any(set(item) != part_keys for item in data['slices']):
            raise ValueError(f'{node.name}: malformed {name} {data}.')
        return data

    if 'concat_view' in node.traits:
        data = trait('concat_view', {'kind', 'axis', 'output', 'slices'}, {'input', 'start', 'extent'})
        parts = tuple(ViewPart(str(s['input']), int(s['start']), int(s['extent'])) for s in data['slices'])
        views = [(str(data['output']), ExecutionView('concat', node.name, int(data['axis']), parts))]
    else:
        data = trait('slice_view', {'kind', 'axis', 'source', 'slices'}, {'output', 'start', 'extent'})
        views = []
        for s in data['slices']:
            part = ViewPart(str(data['source']), int(s['start']), int(s['extent']))
            views.append((str(s['output']), ExecutionView(node.op_type, node.name, int(data['axis']), (part,))))
    if sorted(name for name, _ in views) != sorted(tensor.name for tensor in node.outputs):
        raise ValueError(f'{node.name}: its view names {sorted(name for name, _ in views)}, not its outputs.')
    return views


def _build_execution_values(ctx) -> None:
    """The values the execution graph moves, copied once from the logical graph: its boundary, the
    views folding left without a kernel, and every entry's outputs. From here on transport reads
    these, never the logical tensors."""
    execution = ctx.ir.execution
    execution.values = {}
    execution.graph_inputs = tuple(ctx.ir.logical.input_tensor_names)
    execution.graph_outputs = tuple(ctx.ir.logical.output_tensor_names)
    for name in execution.graph_inputs:
        execution.add_value(ExecutionValue(name))
    for node in ctx.ir.logical:
        if node.is_placeholder and ('slice_view' in node.traits or 'concat_view' in node.traits):
            for name, view in _folded_views(node):
                execution.add_value(ExecutionValue(name, view=view))
    for inst in execution:
        for name in inst.outputs:
            execution.add_value(ExecutionValue(name, producer=inst.name))


class Resolve(AIEPass):
    """Resolve logical nodes into family-owned execution entries."""

    def __init__(self):
        self.name = 'resolve'
        self._registry = get_family_resolver_registry()

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        ctx.ir.logical.verify()
        changed = False
        visited = set()

        ctx.ir.execution.tensor_contracts.clear()

        for node in ctx.ir.logical:
            if node.is_placeholder:
                continue

            resolver = self._registry.get(node.op_type)
            check_io_view(node, ctx.device.generation)

            resolved_directives = dict(node.directives or {})
            resolved_directives['io_route'] = resolve_io_route(node)  # user intents
            resolved_directives['input_contracts'] = _resolved_input_contracts(ctx, node)

            config, variant = resolver.resolve(node, ctx.device, resolved_directives)
            _check_transposed_views(node, config, variant)
            variant.validate_config(node, config, ctx.device)
            ports = variant.build_ports(node, config)
            variant.validate_ports(node, ports, ctx.device)

            # Registered afresh even when unchanged: a later lowering pass may have rewired the old
            # entry (a layout conversion in front of it), and it rewires the new one again.
            previous = ctx.ir.execution.get(node.name)
            same = previous is not None and _same_execution_entry(previous, variant, ports, config)
            inputs = tuple(ExecutionInput(t.name, input_role(node, t.name)) for t in node.inputs if not t.is_parameter)
            outputs = tuple(t.name for t in node.outputs)
            inst = ctx.ir.execution.register(
                node=node,
                variant=variant,
                ports=ports,
                io_route=dict(config.io_route),
                port_views={name: config.io_views[name] for name in (*(item.tensor for item in inputs), *outputs)},
                config=config,
                graph_header=variant.graph_header,
                graph_name=variant.graph_name,
                param_template=variant.param_template,
                inputs=inputs,
                outputs=outputs,
            )
            if same:
                inst.artifacts = previous.artifacts
            _propagate_contracts(ctx, node, inst, config)
            visited.add(node.name)
            changed = changed or not same

        if ctx.ir.execution.prune(visited):
            changed = True
        _build_execution_values(ctx)

        return changed
