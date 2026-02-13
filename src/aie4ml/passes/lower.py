# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Lower hls4ml model graphs to the dedicated AIE IR."""

from __future__ import annotations

from typing import Any, Dict

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
from .utils import attach_default_io_view, is_pointwise_dense


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
        batch_size = int(model.config.get_config_value('AIEConfig', {})['BatchSize'])
        batch_included = bool(ctx.policies.tensors_have_batch)

        def _canon(shape):
            dims = [int(x) for x in shape]
            if batch_included:
                return tuple(dims)
            return tuple([batch_size] + dims)

        if input_var.name not in graph.tensors:
            graph.add_tensor(TensorVar(name=input_var.name, shape=input_var.shape))

        for layer in layers:
            var = model.output_vars[layer.name]
            if var.name not in graph.tensors:
                graph.add_tensor(TensorVar(name=var.name, shape=_canon(var.shape)))

        node_map: Dict[str, OpNode] = {}
        created_nodes = set()

        for layer in layers:
            if layer.class_name == 'Activation' and self._is_identity_activation(layer):
                continue

            node = OpNode(
                name=f'{layer.name}_aie',
                op_type=self._map_op_type(layer),
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

            self._attach_traits(ctx, node, layer)

        return True

    def _collect_metadata(self, layer, node) -> None:
        # legacy metadata (kept for transition)

        meta: Dict[str, Any] = {}

        if layer.class_name == 'Dense' or is_pointwise_dense(layer):
            if layer.class_name == 'Dense':
                n_in = layer.get_attr('n_in')
                n_out = layer.get_attr('n_out')
            else:
                n_in = layer.get_attr('n_chan')
                n_out = layer.get_attr('n_filt')
            if n_in is None or n_out is None:
                raise ValueError(f'{layer.name}: missing n_in/n_out for {layer.class_name}.')
            meta['n_in'] = int(n_in)
            meta['n_out'] = int(n_out)
            meta['use_bias'] = layer.get_attr('bias_data') is not None

        if layer.class_name == 'Activation':
            act = (layer.get_attr('activation', '') or '').lower()
            if act:
                meta['activation'] = act

        meta['layer_class'] = layer.class_name
        if is_pointwise_dense(layer):
            meta['source_class'] = layer.class_name
            meta['layer_class'] = 'Dense'
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
            tensors_have_batch=bool(
                (config.get_config_value('AIEFrontendPolicy', {}) or {}).get('TensorsHaveBatch', False)
            ),
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
        ctx.traits.register(
            TraitDefinition(
                name='io_view',
                dialects=(ctx.device.dialect,),
                fields=('inputs', 'outputs'),
                description='Per-tensor logical-to-physical view mapping for IO/staging.',
            )
        )

    def _attach_traits(self, ctx: AIEBackendContext, node: OpNode, layer) -> None:
        if layer.class_name == 'Dense' or is_pointwise_dense(layer):
            fused = (layer.get_attr('aie_fused_activation', '') or '').lower()
            if fused:
                node.add_trait(TraitInstance('fused_activation', {'activation': fused}))
        attach_default_io_view(node)

    def _is_identity_activation(self, layer) -> bool:
        act = (layer.get_attr('activation', '') or '').lower()
        return act in ('', 'linear', 'identity')

    def _map_op_type(self, layer) -> str:
        if layer.class_name == 'Dense':
            return 'dense'
        if is_pointwise_dense(layer):
            return 'dense'
        return layer.class_name.lower()
