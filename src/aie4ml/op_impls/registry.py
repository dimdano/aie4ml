from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from .base import OpImplSelectionContext, OpImplVariant
from .builtins import register_builtin_op_impls


class OpImplRegistry:
    def __init__(self):
        self._variants = {}

    def register(self, variant: OpImplVariant) -> None:
        self._variants.setdefault(variant.op_type, []).append(variant)

    def variants(self, op_type: str) -> Iterable[OpImplVariant]:
        return self._variants.get(op_type, [])

    def select(self, context: OpImplSelectionContext) -> Optional[OpImplVariant]:
        for variant in self._variants.get(context.node.op_type, []):
            if variant.supports(context):
                return variant
        return None

    def supported_tilings(self, op_type: str, generation: str, query) -> List[Tuple[int, int, int]]:
        candidates = self._variants.get(op_type, [])
        variant = None
        for cand in candidates:
            if cand.supports_generation(generation):
                variant = cand
                break
        if variant is None and candidates:
            variant = candidates[0]
        if variant is None:
            return []
        return variant.tiling_options(generation, query)


_GLOBAL_OP_IMPL_REGISTRY = OpImplRegistry()
register_builtin_op_impls(_GLOBAL_OP_IMPL_REGISTRY)


def get_op_impl_registry() -> OpImplRegistry:
    return _GLOBAL_OP_IMPL_REGISTRY
