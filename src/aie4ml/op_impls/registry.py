from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from .base import OpImplVariant


class OpImplRegistry:
    def __init__(self):
        self._variants = {}

    def register(self, variant: OpImplVariant) -> None:
        self._variants.setdefault(variant.op_type, []).append(variant)

    def variants(self, op_type: str) -> Iterable[OpImplVariant]:
        return self._variants.get(op_type, [])

    def supported_microtilings(self, op_type: str, generation: str, query) -> List[Tuple[int, int, int]]:
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
        return variant.microtiling_options(generation, query)


_GLOBAL_OP_IMPL_REGISTRY = OpImplRegistry()


def get_op_impl_registry() -> OpImplRegistry:
    return _GLOBAL_OP_IMPL_REGISTRY


def register_variant(cls):
    _GLOBAL_OP_IMPL_REGISTRY.register(cls())
    return cls


def select_variant(op_type: str, config, generation: str) -> OpImplVariant:
    precision_query: Dict[str, object] = {key: int(dtype.width) for key, dtype in config.precision.items()}
    for key in ('lhs', 'rhs'):
        c_type = getattr(config.precision.get(key), 'c_type', '') or ''
        if c_type:
            precision_query[f'{key}_c_type'] = c_type

    for variant in _GLOBAL_OP_IMPL_REGISTRY.variants(op_type):
        if not variant.supports_generation(generation):
            continue
        if not variant.supports_io_route(config.io_route):
            continue
        if variant.supports_precisions(precision_query):
            return variant
    raise RuntimeError(f'No implementation variant satisfies resolved {op_type} config.')


from . import families  # noqa: E402,F401
