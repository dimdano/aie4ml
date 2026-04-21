from __future__ import annotations


class FamilyResolverRegistry:
    def __init__(self):
        self._resolvers = {}

    def register(self, op_type: str, resolver) -> None:
        self._resolvers[op_type] = resolver

    def get(self, op_type: str):
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


from . import families  # noqa: E402,F401
