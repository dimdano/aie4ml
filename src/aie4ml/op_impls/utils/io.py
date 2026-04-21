from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ...ir import TraitInstance
from ...ir.graph import STAGING_CONTRACTS, TensorContract


def ensure_io_view(node, generation: str) -> None:
    io_trait = node.traits.get('io_view')
    if io_trait is None:
        io_trait = TraitInstance('io_view', {'inputs': {}, 'outputs': {}})
        node.add_trait(io_trait)
    data = io_trait.data
    data.setdefault('inputs', {})
    data.setdefault('outputs', {})

    gen = (generation or '').upper()
    max_rank = 5 if 'AIE-MLV2' in gen else 4

    def _default_view(rank: int) -> Dict[str, Any]:
        return {
            'layout': 'channels_last',
            'independent_axes': list(range(rank - 1)),
            'buffer_order': list(reversed(range(rank))),
        }

    for direction, tensors in (('inputs', node.inputs), ('outputs', node.outputs)):
        for tensor in tensors:
            rank = len(tuple(int(x) for x in tensor.shape))
            if rank > max_rank:
                raise ValueError(f'{node.name}: tensor rank {rank} exceeds max {max_rank} for {generation}.')
            if tensor.name not in data[direction]:
                data[direction][tensor.name] = _default_view(rank)


def resolve_io_route(node) -> Dict[str, Any]:
    route = {'inputs': {}, 'outputs': {}}
    for tensor in node.inputs:
        route['inputs'][tensor.name] = 'auto'
    for tensor in node.outputs:
        route['outputs'][tensor.name] = 'auto'

    user = node.directives.get('io_route', {})
    for direction in ('inputs', 'outputs'):
        if isinstance(user.get(direction), dict):
            route[direction].update(user[direction])
    return route


def resolve_input_contract(
    input_contracts: Dict[str, TensorContract],
    tensor_names: Sequence[str],
    default: str = 'outer',
) -> tuple[str, Dict[str, str]]:
    """Choose a multi-input staging contract from propagated producer contracts.

    Returns (contract, io_route_patches). Inputs whose contract differs from the
    chosen one are patched to 'memtile'. The first known input contract wins.
    """

    found = {name: input_contracts[name] for name in tensor_names if name in input_contracts}
    if not found:
        return default, {}

    primary_name = next(name for name in tensor_names if name in found)
    contract = found[primary_name].contract

    if contract not in STAGING_CONTRACTS:
        raise ValueError(
            f'Producer emitted unknown staging contract {contract!r}; ' f'expected one of {sorted(STAGING_CONTRACTS)}.'
        )

    patches: Dict[str, str] = {name: 'memtile' for name, tc in found.items() if tc.contract != contract}
    return contract, patches


def normalized_staging(desc: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if desc is None:
        return None
    data = {k: v for k, v in desc.items() if k not in ('access', 'boundary_dimension')}
    if 'io_boundary_dimension' in data and 'boundary_dimension' not in data:
        data['boundary_dimension'] = data['io_boundary_dimension']
    return data


def view_shape(node, tensor, direction: str) -> List[int]:
    logical = [int(x) for x in tensor.shape]
    view = view_layout(node, tensor, direction)
    perm = view.get('perm')
    if perm is None:
        return logical
    if sorted(perm) != list(range(len(logical))):
        raise ValueError(f'{node.name}: invalid io_view perm {perm} for rank {len(logical)}.')
    return [int(logical[i]) for i in perm]


def view_layout(node, tensor, direction: str) -> Dict[str, Any]:
    io_trait = node.traits.get('io_view')
    if io_trait is None:
        raise ValueError(f'{node.name}: missing io_view trait.')
    return io_trait.data[direction][tensor.name]
