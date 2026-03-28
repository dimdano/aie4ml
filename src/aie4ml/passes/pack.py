# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Pass to pack kernel artifacts into tiled layouts for AIE mmul kernels."""

from hls4ml.model.optimizer.optimizer import ModelOptimizerPass

from ..ir import get_backend_context
from ..op_impls.base import OpImplVariant


class PackKernelArtifacts(ModelOptimizerPass):
    """
    Packs kernel-resident tensors into variant-specific tiled layouts
    required by AIE mmul-based kernels.
    """

    def __init__(self):
        self.name = 'pack_kernel_artifacts'

    def transform(self, model):
        ctx = get_backend_context(model)
        changed = False

        for inst in ctx.ir.execution:
            if type(inst.variant).pack is OpImplVariant.pack:
                continue

            if 'packed_weights' in inst.artifacts:
                continue

            packed = inst.variant.pack(inst)

            if packed:
                inst.artifacts.update(packed)
                changed = True

        return changed
