# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Resolve per-layer AIE attributes using policy-driven resolver pipelines."""

from __future__ import annotations

import logging

from hls4ml.model.optimizer.optimizer import ModelOptimizerPass

from ..ir import ResolvedAttributes, get_backend_context
from ..op_impls import OpImplSelectionContext, get_op_impl_registry
from .resolve_registry import LayerResolveContext, get_layer_policy, merge_config_layers

log = logging.getLogger(__name__)


def _quant_meta(node) -> dict:
    """Build the quant precision dict from a node's tensor inputs/outputs."""
    meta: dict = {}
    if node.inputs:
        meta['input_precision'] = node.inputs[0].precision
    if node.outputs:
        meta['output_precision'] = node.outputs[0].precision
    if len(node.inputs) > 1 and node.inputs[1].is_parameter:
        meta['weight_precision'] = node.inputs[1].precision
    if len(node.inputs) > 2 and node.inputs[2].is_parameter:
        meta['bias_precision'] = node.inputs[2].precision
    return meta


def resolve_aie_attributes(model, ctx, node) -> ResolvedAttributes:
    """Run the registered resolver pipeline for the given IR node."""

    layer_class = node.metadata.get('layer_class')
    policy = get_layer_policy(layer_class)
    layer_name = node.metadata['source_layer']
    merged_cfg = merge_config_layers(model.config.config, layer_name, layer_class)
    attributes = ResolvedAttributes()

    context = LayerResolveContext(
        model=model,
        backend_ctx=ctx,
        node=node,
        layer_name=layer_name,
        layer_class=layer_class,
        policy=policy,
        config=merged_cfg,
        quant=_quant_meta(node),
        device=ctx.device,
        global_config=model.config.get_config_value('AIEConfig', {}) or {},
        attributes=attributes,
    )

    for resolver in policy.resolvers:
        resolver(context)

    if ctx.policies.pack:
        pack_policy = ctx.policies.pack
        attributes.pack.setdefault(
            'policy',
            dict(pack_policy) if isinstance(pack_policy, dict) else pack_policy,
        )
    if ctx.policies.cache:
        cache_policy = ctx.policies.cache
        attributes.pack.setdefault(
            'cache',
            dict(cache_policy) if isinstance(cache_policy, dict) else cache_policy,
        )

    for namespace in policy.namespaces:
        attr_value = getattr(attributes, namespace, None)
        if attr_value is None or (isinstance(attr_value, dict) and not attr_value):
            raise RuntimeError(
                f'{layer_name}: resolver pipeline did not populate required attribute namespace "{namespace}".'
            )

    return attributes


class Resolve(ModelOptimizerPass):
    """Merge global/local AIE config and derive per-layer resolved attributes."""

    def __init__(self):
        self.name = 'resolve'
        self._registry = get_op_impl_registry()

    def transform(self, model):
        ctx = get_backend_context(model)
        changed = False
        visited = set()
        for node in ctx.ir.logical:
            if node.metadata['layer_class'] == 'Input':
                continue

            resolved = resolve_aie_attributes(model, ctx, node)
            selection_ctx = OpImplSelectionContext(
                node=node,
                attributes=resolved,
                device_generation=ctx.device.generation,
                metadata=dict(node.metadata),
            )
            variant = self._registry.select(selection_ctx)
            if variant is None:
                raise RuntimeError(f'{node.name}: no implementation variant satisfies resolved attributes.')

            kernel_cfg = variant.build_config(selection_ctx)
            inst = ctx.ir.execution.get(node.name)
            if inst is not None:
                same_variant = inst.variant.variant_id == variant.variant_id
                same_cfg = inst.config == kernel_cfg
                if same_variant and same_cfg:
                    visited.add(node.name)
                    continue

            ctx.ir.execution.register(node, variant, kernel_cfg)
            visited.add(node.name)
            changed = True

        if ctx.ir.execution.prune(visited):
            changed = True

        return changed
