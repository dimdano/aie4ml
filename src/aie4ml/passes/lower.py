# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Lower hls4ml model graphs to the dedicated AIE IR."""

from __future__ import annotations

from typing import Dict

from hls4ml.model.optimizer.optimizer import ModelOptimizerPass

from ..device_catalog import load_device_catalog
from ..ir import (
    BackendPolicies,
    LogicalIR,
    OpNode,
    TensorVar,
    TraitDefinition,
    TraitInstance,
    ensure_backend_context,
)
from ..ir.context import AIEBackendContext, DeviceSpec


class LowerToAieIr(ModelOptimizerPass):
    """Build the shared IR graph from the frontend model."""

    def __init__(self):
        self.name = 'lower_to_aie_ir'

    def transform(self, model) -> bool:
        ctx = ensure_backend_context(model, lambda: self._create_context(model))
        ctx.reset_ir()

        graph: LogicalIR = ctx.ir.logical
        layers = list(model.get_layers())
        input_var = model.get_input_variables()[0]

        for layer in layers:
            var = model.output_vars[layer.name]
            if var.name not in graph.tensors:
                graph.add_tensor(TensorVar(name=var.name, shape=tuple(var.shape)))

        node_map: Dict[str, OpNode] = {}
        created_nodes = set()

        for layer in layers:

            node = OpNode(
                name=f'{layer.name}_aie',
                op_type=layer.class_name.lower(),
                dialect=ctx.device.dialect,
            )
            self._collect_metadata(layer, node)

            var = model.output_vars[layer.name]
            tv = graph.tensors[var.name]
            tv.producer = node
            node.outputs.append(tv)
            graph.add_node(node)
            node_map[layer.name] = node
            created_nodes.add(layer.name)

        for layer in layers:
            if layer.name not in created_nodes:
                continue
            node = node_map[layer.name]
            if layer.class_name.lower() == 'input':
                continue

            for src in layer.inputs:
                if src == 'input':
                    var = input_var
                else:
                    var = model.output_vars[src]

                tv = graph.tensors[var.name]
                node.inputs.append(tv)
                tv.consumers.append(node)

            self._attach_traits(node, layer)

        return True

    def _collect_metadata(self, layer, node) -> None:
        # legacy metadata (kept for transition)

        meta: Dict[str, Any] = {}

        if layer.class_name == 'Dense':
            meta['n_in'] = int(layer.get_attr('n_in'))
            meta['n_out'] = int(layer.get_attr('n_out'))
            meta['use_bias'] = layer.get_attr('bias_data') is not None

        if layer.class_name == 'Activation':
            act = (layer.get_attr('activation', '') or '').lower()
            if act:
                meta['activation'] = act

        meta['layer_class'] = layer.class_name
        meta['source_layer'] = layer.name

        if meta:
            node.metadata.update(meta)

    def _create_context(self, model) -> AIEBackendContext:
        config = model.config
        aie_cfg = config.get_config_value('AIEConfig', {}) or {}
        part_name = aie_cfg.get('Device') or aie_cfg.get('Part') or config.get_config_value('Part') or 'unknown_part'

        catalog = load_device_catalog()
        device_entry = catalog.get(part_name, {}) or catalog.get(part_name.lower(), {})
        merged = dict(device_entry)
        merged.update(aie_cfg)

        if 'Generation' not in merged:
            merged['Generation'] = device_entry.get('Generation', '')

        device = DeviceSpec.from_config(part_name, merged)
        policies = BackendPolicies(
            fusion=config.get_config_value('AIEFusionPolicy', {}) or {},
            decomposition=config.get_config_value('AIEDecompositionPolicy', {}) or {},
            pack=config.get_config_value('AIEPackPolicy', {}) or {},
            cache=config.get_config_value('AIECachePolicy', {}) or {},
        )

        ctx = AIEBackendContext(device=device, policies=policies)
        self._register_default_traits(ctx)
        return ctx

    @staticmethod
    def _register_default_traits(ctx: AIEBackendContext) -> None:
        ctx.traits.register(
            TraitDefinition(
                name='fused_activation',
                dialects=(ctx.device.dialect,),
                fields=('activation',),
                description='Indicates that an activation has been fused into the producer op.',
            )
        )

    def _attach_traits(self, node: OpNode, layer) -> None:
        if layer.class_name == 'Dense':
            fused = (layer.get_attr('aie_fused_activation', '') or '').lower()
            if fused:
                node.add_trait(TraitInstance('fused_activation', {'activation': fused}))
