# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Fold per-channel scale nodes into their predecessor Dense weight precision."""

import numpy as np
from hls4ml.model.optimizer.optimizer import ModelOptimizerPass

from ..aie_types import FloatIntent, QuantIntent
from ..ir import get_backend_context


class FoldApplyAlpha(ModelOptimizerPass):
    """Fold output scale into a preceding Dense when legal.

    For applyalpha nodes with constant per-channel scale, this pass updates the
    producer Dense quant intent and bias, then removes the scale node.

    This is mainly a graph simplification/optimization pass. It is only bit-exact
    when the incoming weight data already matches the frontend quantization grid,
    or that grid has been reconstructed exactly before folding.
    """

    def __init__(self):
        self.name = 'fold_apply_alpha'

    def transform(self, model):
        ctx = get_backend_context(model)
        graph = ctx.ir.logical
        changed = False

        for node in list(graph.nodes):
            if node.op_type != 'applyalpha':
                continue

            scale = np.asarray(node.metadata['scale'], dtype=np.float64)
            p_min = int(np.round(-np.log2(scale.max())))

            dense_node = node.inputs[0].producer

            weight_tv = dense_node.inputs[1]
            old_w = weight_tv.precision
            if isinstance(old_w, FloatIntent):
                continue
            weight_tv.precision = QuantIntent(
                width=old_w.width,
                frac=old_w.frac + p_min,
                signed=old_w.signed,
                rounding=old_w.rounding,
                saturation=old_w.saturation,
            )

            if dense_node.metadata.get('use_bias') and len(dense_node.inputs) > 2:
                bias_tv = dense_node.inputs[2]
                bias_tv.data = np.asarray(bias_tv.data, dtype=np.float64) * scale

            graph.remove_node(node, mode='contract')
            changed = True

        return changed
