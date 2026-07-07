# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Activation ops: Relu."""

from __future__ import annotations

from ..context import OnnxImportContext
from ..registry import onnx_handler


@onnx_handler('Relu')
def _relu(ctx: OnnxImportContext, node, node_name: str, directives: dict) -> None:
    if len(node.input) != 1:
        raise ValueError(f'{node_name}: Relu must have exactly 1 input.')
    src = ctx.source_for(node.input[0], node_name)
    out_name = node.output[0]
    ctx.emit(
        'activation',
        node_name,
        inputs=[src],
        outputs=[(out_name, ctx.output_shape(out_name, node_name), src.precision)],
        roles=['lhs'],
        metadata={'activation': 'relu', 'layer_class': 'Activation', 'source_layer': node_name},
        directives=directives,
    )
