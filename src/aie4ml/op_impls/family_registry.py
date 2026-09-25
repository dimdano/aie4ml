from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Tuple

from .utils import requested_port_kind

if TYPE_CHECKING:
    from .base import OpImplVariant


class FamilyResolver:
    """Structural validator, variant dispatcher and capability record for one op type; passes ask
    capabilities instead of naming op types."""

    op_type: ClassVar[str] = ''
    supported_fusions: ClassVar[frozenset] = frozenset()  # epilogues folded into the kernel
    supported_output_views: ClassVar[frozenset] = frozenset()  # OUTPUT_VIEWS written directly

    def spatial_access(self, _node: Any):
        """The 2-D window read around each output pixel (sizes its producer's frame), or None."""
        return None

    def reorder_reduction_rows(self, node: Any, tensor: Any, _order) -> None:
        """Adopt a folded view's permutation of the rows this op reduces over into its constants."""
        raise NotImplementedError(
            f'{node.name}: {self.op_type} cannot adopt a reordered {tensor.name!r}; its constants '
            'assume the original row order.'
        )

    def validate_structure(self, _node: Any, _device: Any) -> None:
        raise NotImplementedError

    def resolve(self, node: Any, device: Any, directives: Optional[Dict[str, Any]] = None) -> Tuple[Any, OpImplVariant]:
        from .registry import get_op_impl_registry

        self.validate_structure(node, device)
        for variant in get_op_impl_registry().candidates(self.op_type):
            if variant.matches(node, device):
                config = variant.resolve(node, device, directives)
                return config, variant
        raise ValueError(
            f'{node.name}: no {self.op_type} variant matches '
            f'(generation={device.generation!r}, ports={requested_port_kind(node)!r}).'
        )


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

    def find(self, op_type: str) -> Optional[FamilyResolver]:
        """Like `get`, but None for an op no family implements (a view)."""
        return self._resolvers.get(op_type)


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
