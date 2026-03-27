# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""ForceFloatMode: replace all IR tensor precisions with FloatIntent.

Development/testing pass that lets the float path be exercised through
the hls4ml frontend before a native float frontend exists.
Activated by setting AIEConfig.ComputeDtype to 'bfloat16' or 'float32'.
"""

from hls4ml.model.optimizer.optimizer import ModelOptimizerPass

from ..aie_types import FloatFormat, FloatIntent, QuantIntent
from ..ir import get_backend_context

_FORMAT_MAP = {
    'bfloat16': (FloatFormat.BF16, 16),
    'float32': (FloatFormat.FP32, 32),
    'float': (FloatFormat.FP32, 32),
}


class ForceFloatMode(ModelOptimizerPass):
    """Replace every QuantIntent precision in the logical IR with FloatIntent."""

    def __init__(self):
        self.name = 'force_float_mode'

    def transform(self, model) -> bool:
        aie_cfg = model.config.get_config_value('AIEConfig', {}) or {}
        fmt_str = (aie_cfg.get('ComputeDtype') or '').lower()
        if not fmt_str:
            return False

        if fmt_str not in _FORMAT_MAP:
            raise ValueError(f'ComputeDtype={fmt_str!r} is not recognised. ' f'Use one of: {list(_FORMAT_MAP)}')

        fmt, width = _FORMAT_MAP[fmt_str]
        float_intent = FloatIntent(width=width, format=fmt)

        ctx = get_backend_context(model)
        changed = False
        for tv in ctx.ir.logical.tensors.values():
            if isinstance(tv.precision, QuantIntent):
                tv.precision = float_intent
                changed = True

        return changed
