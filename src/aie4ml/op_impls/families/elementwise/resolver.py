from __future__ import annotations

import math
from dataclasses import dataclass

from ....aie_types import FloatIntent
from ....ir import input_tensor_for_role
from ...family_registry import family_resolver
from ...registry import select_variant as _select_variant
from ...utils import align_up, build_tensor_view, build_tensor_view_from_staging, ceildiv
from ...utils.io import resolve_input_contract, view_shape
from ...utils.precision import (
    aie_rounding_token,
    infer_accumulator_tag,
    resolve_accumulator_output_shift,
    resolve_exact_storage_dtype,
    storage_bytes_for_spec,
)
from .common import elementwise_vec_size
from .config import AddConfig, ElementwiseParallelismConfig


@dataclass(frozen=True)
class ElementwiseTiling:
    cas_num: int
    tile_outer_raw: int
    full_outer: int
    tile_inner_raw: int
    tile_inner: int


def resolve_elementwise_parallelism(
    *,
    contract: str,
    outer_prefix: int,
    last_outer: int,
    raw_inner: int,
    full_inner: int,
    elem_bytes: int,
    device,
    requested_cas_num: int | None = None,
) -> ElementwiseTiling:
    """Resolve port tiling for an elementwise op.

    For 'outer' contract the ports partition the outer dimension; for 'inner'
    they partition the inner (feature) dimension.  Both must fit one bank per port.
    """
    max_rows = max(1, int(device.rows))
    bank_bytes = int(device.bank_mem_bytes)
    compacted_outer = int(outer_prefix) * int(last_outer)

    if contract == 'inner':
        if requested_cas_num is None:
            raise ValueError(
                f'{contract!r} contract requires port_count from the producer TensorContract; '
                'check that the producer op is resolved before this one.'
            )
        cas_num = int(requested_cas_num)
        if cas_num < 1 or cas_num > max_rows:
            raise ValueError(f'inner contract: cas_num={cas_num} invalid for rows={device.rows}.')
        if full_inner % cas_num != 0:
            raise ValueError(f'inner contract: full_inner={full_inner} must be divisible by cas_num={cas_num}.')
        tile_inner_raw = ceildiv(int(raw_inner), cas_num)
        tile_inner = int(full_inner) // cas_num
        tile_bytes = int(compacted_outer) * int(tile_inner) * max(1, int(elem_bytes))
        if tile_bytes > bank_bytes:
            raise ValueError(f'inner contract tile uses {tile_bytes}B, exceeds one {bank_bytes}B bank.')
        return ElementwiseTiling(
            cas_num=cas_num,
            tile_outer_raw=int(last_outer),
            full_outer=int(last_outer),
            tile_inner_raw=tile_inner_raw,
            tile_inner=tile_inner,
        )

    # 'outer' contract: search for a cas_num that fits one bank.
    cas_candidates = [int(requested_cas_num)] if requested_cas_num is not None else range(1, max_rows + 1)
    if requested_cas_num is not None:
        cas_num = int(requested_cas_num)
        if cas_num < 1 or cas_num > min(max_rows, int(last_outer)):
            raise ValueError(
                f'outer contract: cas_num={cas_num} invalid for ' f'last_outer={last_outer} and rows={device.rows}.'
            )
    for cas_num in cas_candidates:
        tile_outer_raw = ceildiv(int(last_outer), int(cas_num))
        full_outer = int(tile_outer_raw * cas_num)
        compacted_tile_outer = int(outer_prefix) * int(tile_outer_raw)
        tile_bytes = int(compacted_tile_outer) * int(full_inner) * max(1, int(elem_bytes))
        if tile_bytes <= bank_bytes:
            return ElementwiseTiling(
                cas_num=int(cas_num),
                tile_outer_raw=int(tile_outer_raw),
                full_outer=int(full_outer),
                tile_inner_raw=int(raw_inner),
                tile_inner=int(full_inner),
            )

    raise ValueError(
        f'outer contract: no legal parallelism fits one bank '
        f'(outer_prefix={outer_prefix}, last_outer={last_outer}, full_inner={full_inner}, '
        f'elem_bytes={elem_bytes}, rows={device.rows}, bank_bytes={bank_bytes}).'
    )


def _select_preserved_staging(
    tensor_names: tuple[str, str],
    input_contracts,
) -> tuple[tuple[dict, ...] | None, dict[str, str]]:
    primary_name = next((name for name in tensor_names if name in input_contracts), None)
    if primary_name is None:
        return None, {}

    primary = input_contracts[primary_name].port_staging
    patches = {
        name: 'memtile'
        for name in tensor_names
        if name in input_contracts and input_contracts[name].port_staging != primary
    }
    return primary, patches


@family_resolver('add')
class AddResolver:
    op_type = 'add'

    def resolve(self, node, device, directives=None) -> AddConfig:
        io_route = dict((directives or {}).get('io_route', {}))

        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')

        input_contracts = (directives or {}).get('input_contracts', {})
        staging_contract, conflict_patches = resolve_input_contract(
            input_contracts,
            [lhs_tensor.name, rhs_tensor.name],
        )
        preserved_staging, staging_patches = _select_preserved_staging(
            (lhs_tensor.name, rhs_tensor.name),
            input_contracts,
        )
        route_patches = dict(conflict_patches)
        route_patches.update(staging_patches)
        if route_patches:
            inputs_route = dict(io_route.get('inputs', {}))
            inputs_route.update(route_patches)
            io_route = dict(io_route)
            io_route['inputs'] = inputs_route

        lhs_shape = tuple(int(x) for x in view_shape(node, lhs_tensor, 'inputs'))
        rhs_shape = tuple(int(x) for x in view_shape(node, rhs_tensor, 'inputs'))
        if lhs_shape != rhs_shape:
            raise ValueError(
                f'{node.name}: elementwise Add requires exact-shape inputs, got {lhs_shape} and {rhs_shape}.'
            )
        if len(lhs_shape) < 2:
            raise ValueError(f'{node.name}: elementwise Add requires rank >=2 tensors, got {len(lhs_shape)}.')

        precision = {
            'lhs': resolve_exact_storage_dtype(lhs_tensor.precision, namespace='lhs', layer_name=node.name),
            'rhs': resolve_exact_storage_dtype(rhs_tensor.precision, namespace='rhs', layer_name=node.name),
            'output': resolve_exact_storage_dtype(node.outputs[0].precision, namespace='output', layer_name=node.name),
        }
        if any(isinstance(t.precision, FloatIntent) for t in (lhs_tensor, rhs_tensor, node.outputs[0])):
            if not all(isinstance(t.precision, FloatIntent) for t in (lhs_tensor, rhs_tensor, node.outputs[0])):
                raise ValueError(f'{node.name}: elementwise Add requires lhs/rhs/output to use the same float type.')
        if precision['lhs'] != precision['rhs'] or precision['lhs'] != precision['output']:
            raise ValueError(f'{node.name}: elementwise Add requires lhs/rhs/output to use the same storage type.')

        is_float = isinstance(lhs_tensor.precision, FloatIntent)

        vec_size = elementwise_vec_size(precision['lhs'], device)
        raw_inner = int(lhs_shape[-1])

        if preserved_staging is not None:
            port0 = preserved_staging[0]
            cas_num = len(preserved_staging)
            io_views = {
                tensor.name: build_tensor_view_from_staging(node, tensor, 'inputs', port0) for tensor in node.inputs
            }
            io_views.update(
                {tensor.name: build_tensor_view_from_staging(node, tensor, 'outputs', port0) for tensor in node.outputs}
            )
        else:
            outer_prefix = int(math.prod(lhs_shape[:-2])) if len(lhs_shape) > 2 else 1
            last_outer = int(lhs_shape[-2])
            full_inner = align_up(raw_inner, vec_size)
            parallel_cfg = (directives or {}).get('parallelism', {}) or {}
            tiling = resolve_elementwise_parallelism(
                contract=staging_contract,
                outer_prefix=outer_prefix,
                last_outer=last_outer,
                raw_inner=raw_inner,
                full_inner=full_inner,
                elem_bytes=storage_bytes_for_spec(precision['lhs']),
                device=device,
                requested_cas_num=parallel_cfg.get('cas_num'),
            )
            cas_num = int(tiling.cas_num)

            io_views = {}
            for tensor in node.inputs:
                io_views[tensor.name] = build_tensor_view(
                    node,
                    tensor,
                    'inputs',
                    full_inner=full_inner,
                    tile_inner=tiling.tile_inner,
                    tile_inner_raw=tiling.tile_inner_raw,
                    full_outer=tiling.full_outer,
                    tile_outer=tiling.tile_outer_raw,
                    tile_outer_raw=tiling.tile_outer_raw,
                )
            for tensor in node.outputs:
                io_views[tensor.name] = build_tensor_view(
                    node,
                    tensor,
                    'outputs',
                    full_inner=full_inner,
                    tile_inner=tiling.tile_inner,
                    tile_inner_raw=tiling.tile_inner_raw,
                    full_outer=tiling.full_outer,
                    tile_outer=tiling.tile_outer_raw,
                    tile_outer_raw=tiling.tile_outer_raw,
                )

        if is_float:
            shift = 0
            accumulator_tag = 'accfloat'
            rounding_mode = 'conv_even'
        else:
            shift = resolve_accumulator_output_shift(lhs_tensor.precision, node.outputs[0].precision)
            accumulator_tag = infer_accumulator_tag(device, precision['lhs'], precision['rhs'], precision.get('acc'))
            rounding_mode = aie_rounding_token(precision['output'])

        return AddConfig(
            precision=precision,
            parallelism=ElementwiseParallelismConfig(cas_num=int(cas_num)),
            vec_size=vec_size,
            io_views=io_views,
            io_route=io_route,
            shift=shift,
            accumulator_tag=accumulator_tag,
            rounding_mode=rounding_mode,
            staging_contract=staging_contract,
            preserved_staging=preserved_staging,
        )

    def select_variant(self, config: AddConfig, generation: str):
        return _select_variant(self.op_type, config, generation)
