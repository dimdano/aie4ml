from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Tuple

if TYPE_CHECKING:
    from .base import OpImplVariant


class FamilyResolver:
    """Thin structural validator + variant dispatcher for one op type."""

    op_type: ClassVar[str] = ''

    def validate_structure(self, _node: Any, _device: Any) -> None:
        raise NotImplementedError

    def resolve(self, node: Any, device: Any, directives: Optional[Dict[str, Any]] = None) -> Tuple[Any, OpImplVariant]:
        from .registry import get_op_impl_registry

        self.validate_structure(node, device)
        for variant in get_op_impl_registry().candidates(self.op_type):
            if variant.matches(node, device):
                config = variant.resolve(node, device, directives)
                return config, variant
        raise ValueError(f'{node.name}: no {self.op_type} variant matches ' f'(generation={device.generation!r}).')


class FamilyResolverRegistry:
    def __init__(self):
        self._resolvers: dict[str, FamilyResolver] = {}

    def register(self, op_type: str, resolver: FamilyResolver) -> None:
        self._resolvers[op_type] = resolver

    def get(self, op_type: str) -> FamilyResolver:
        resolver = self._resolvers.get(op_type)
        if resolver is None:
            raise NotImplementedError(f'No family resolver registered for op_type={op_type!r}.')
        return resolver


_GLOBAL_FAMILY_RESOLVER_REGISTRY = FamilyResolverRegistry()


def get_family_resolver_registry() -> FamilyResolverRegistry:
    return _GLOBAL_FAMILY_RESOLVER_REGISTRY


def family_resolver(*op_types: str):
    def decorator(cls):
        instance = cls()
        for op_type in op_types:
            _GLOBAL_FAMILY_RESOLVER_REGISTRY.register(str(op_type), instance)
        return cls

    return decorator
