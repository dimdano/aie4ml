# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from ..ir import get_backend_context
from .base import AIEPass


class FoldBias(AIEPass):
    """Fold a trailing `add(dense_output, const)` into the dense op's bias.

    The ONNX frontend emits bias as a separate `add` (both for a standalone
    MatMul+Add and for Gemm's C input); this pass is the single place that
    fuses it, keeping the frontend free of fusion decisions.
    """

    def __init__(self):
        self.name = 'fold_bias'

    def transform(self, model_or_ctx):
        ctx = get_backend_context(model_or_ctx)
        graph = ctx.ir.logical
        changed = False

        for add in list(graph.nodes):
            if add.op_type != 'add' or len(add.inputs) != 2 or len(add.outputs) != 1:
                continue
            dense_out, bias = _match_dense_bias(add.inputs[0], add.inputs[1])
            if dense_out is None:
                continue

            dense = dense_out.producer
            n_out = int(dense.metadata['n_out'])
            bias_size = int(np.asarray(bias.data).size)
            if bias_size != n_out:
                raise ValueError(f'{add.name}: dense bias Add requires exactly {n_out} elements, got {bias_size}.')
            if len(dense_out.consumers) != 1:
                raise ValueError(
                    f'{add.name}: cannot fold bias because dense output {dense_out.name!r} '
                    f'has {len(dense_out.consumers)} consumers.'
                )

            _fold(graph, dense, dense_out, bias, add)
            changed = True

        return changed


def _match_dense_bias(lhs, rhs):
    if _is_biasless_dense(lhs) and rhs.is_parameter:
        return lhs, rhs
    if _is_biasless_dense(rhs) and lhs.is_parameter:
        return rhs, lhs
    return None, None


def _is_biasless_dense(tensor) -> bool:
    producer = tensor.producer
    return producer is not None and producer.op_type == 'dense' and not producer.metadata.get('use_bias')


def _fold(graph, dense, dense_out, bias, add) -> None:
    y = add.outputs[0]

    dense.inputs.append(bias)
    bias.consumers = [c for c in bias.consumers if c is not add]
    bias.consumers.append(dense)
    dense.metadata['use_bias'] = True
    dense.metadata['input_roles'] = ['lhs', 'rhs', 'bias']
    dense.roles[bias.name] = 'bias'

    dense.outputs = [y]
    y.producer = dense
    graph.tensors.pop(dense_out.name, None)

    graph.nodes.remove(add)
    add.inputs.clear()
    add.outputs.clear()
