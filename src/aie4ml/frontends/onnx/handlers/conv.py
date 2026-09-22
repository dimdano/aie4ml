# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Conv lowering: ONNX NCHW semantics onto the canonical NHWC conv2d contract."""

from __future__ import annotations

import numpy as np

from ..context import OnnxImportContext
from ..registry import onnx_handler
from ..utils import attr

NCHW_OVER_NHWC = (0, 3, 1, 2)
"""How an ONNX NCHW value views a canonical NHWC tensor."""


@onnx_handler('Conv')
def _conv(ctx: OnnxImportContext, node, node_name: str, directives: dict) -> None:
    """Translate ONNX Conv into the canonical conv2d contract.

    Only what translation itself needs is checked here. What the operation must satisfy to make
    sense is the family's contract, and what a kernel can actually run is its variant's.
    """
    if len(node.input) not in (2, 3):
        raise ValueError(f'{node_name}: Conv must have 2 or 3 inputs.')
    x_name, w_name = node.input[:2]
    # ONNX Conv is NCHW; aie4ml activations are canonically NHWC. Export channels-last
    # (input [N,H,W,C] -> Transpose(0,3,1,2) -> Conv ...) so the transpose folds into the view.
    ctx.require_order(x_name, NCHW_OVER_NHWC, node_name)
    x = ctx.source_for(x_name, node_name)
    weight = ctx.parameter_source_for(w_name, node_name)
    if len(x.shape) != 4 or len(weight.shape) != 4:
        raise ValueError(
            f'{node_name}: Conv takes a rank-4 activation and rank-4 weights, got {x.shape} and {weight.shape}.'
        )
    auto_pad = attr(node, 'auto_pad', b'NOTSET')
    auto_pad = auto_pad.decode() if isinstance(auto_pad, bytes) else str(auto_pad)
    if auto_pad != 'NOTSET':
        raise NotImplementedError(
            f'{node_name}: auto_pad={auto_pad} leaves the padding to shape inference; re-export with explicit pads.'
        )

    cout, _cin_g, kh, kw = (int(d) for d in weight.shape)
    metadata = {
        'kernel_shape': tuple(int(k) for k in attr(node, 'kernel_shape', [kh, kw])),
        'strides': tuple(int(s) for s in attr(node, 'strides', [1, 1])),
        'dilations': tuple(int(d) for d in attr(node, 'dilations', [1, 1])),
        'pads': tuple(int(p) for p in attr(node, 'pads', [0, 0, 0, 0])),
        'groups': int(attr(node, 'group', 1)),
        'layer_class': 'Conv2D',
        'source_class': 'Conv',
        'source_layer': node_name,
    }

    # Canonical conv weights: [kh, kw, Cin/groups, Cout]. The variant expands the groups when it
    # packs, so a future depthwise kernel can consume the compact form unchanged.
    w_canonical = np.asarray(weight.data, dtype=np.float64).transpose(2, 3, 1, 0)
    w_param = ctx.param_tensor(f'{node_name}_weight', w_canonical, weight.precision)

    out_name = node.output[0]
    batch, onnx_cout, out_h, out_w = (int(d) for d in ctx.output_shape(out_name, node_name))
    if onnx_cout != cout:
        raise ValueError(f'{node_name}: ONNX output has {onnx_cout} channels but the weights produce {cout}.')
    out_shape = (batch, out_h, out_w, cout)  # the same tensor, in canonical order

    if len(node.input) == 2:
        ctx.emit(
            'conv2d',
            node_name,
            inputs=[x, w_param],
            outputs=[(out_name, out_shape, None)],
            roles=['lhs', 'rhs'],
            metadata=metadata,
            directives=directives,
        )
        ctx.set_order(out_name, NCHW_OVER_NHWC, node_name)
        return

    # Conv with bias: conv2d -> add(bias); FoldBias fuses the add into the conv.
    prebias_name = f'{node_name}_prebias'
    ctx.mirror_precision(prebias_name, out_name)
    ctx.emit(
        'conv2d',
        node_name,
        inputs=[x, w_param],
        outputs=[(prebias_name, out_shape, None)],
        roles=['lhs', 'rhs'],
        metadata=metadata,
        directives=directives,
    )
    bias_tensor = ctx.parameter_source_for(node.input[2], node_name)
    bias_data = np.asarray(bias_tensor.data, dtype=np.float64).reshape(-1)
    if int(bias_data.size) != cout:
        raise ValueError(f'{node_name}: Conv bias must contain exactly {cout} elements.')
    bias_param = ctx.param_tensor(f'{node_name}_bias', bias_data, bias_tensor.precision)
    ctx.emit(
        'add',
        f'{node_name}_bias',
        inputs=[ctx.value_tensors[prebias_name], bias_param],
        outputs=[(out_name, out_shape, None)],
        roles=['lhs', 'rhs'],
        metadata={'layer_class': 'Add', 'source_class': 'Conv', 'source_layer': node_name},
        directives={},
    )
    ctx.set_order(out_name, NCHW_OVER_NHWC, node_name)
    ctx.set_order(prebias_name, NCHW_OVER_NHWC, node_name)
