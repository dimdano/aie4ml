from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ....aie_types import AIEDataType, FloatIntent
from ....ir import input_role, input_tensor_for_role
from ...family_registry import family_resolver
from ...registry import get_op_impl_registry
from ...registry import select_variant as _select_variant
from ...utils import TensorView, align_up, build_tensor_view, ceildiv
from ...utils.io import view_shape
from ...utils.precision import (
    aie_rounding_token,
    element_bytes,
    infer_accumulator_tag,
    resolve_exact_storage_dtype,
    to_quant_intent,
)
from .common import microtile_key
from .config import (
    DenseConfig,
    DenseFlags,
    MatmulConfig,
    MatmulFlags,
    MatmulMicrotileConfig,
    MatmulParallelismConfig,
)


@dataclass(frozen=True)
class MatmulTiling:
    cas_num: int
    cas_length: int
    tile_inner_lhs_raw: int
    tile_inner_lhs: int
    tile_inner_rhs_raw: int
    tile_inner_rhs: int


def _bank_capacity_bytes(device: Any) -> int:
    return max(1, int(getattr(device, 'bank_mem_bytes', 0) or 1))


def _rhs_stack_overhead_bytes(op_type: str) -> int:
    return 1024 if op_type == 'matmul' else 0


def _tile_bank_usage(
    *,
    op_type: str,
    device: Any,
    full_outer: int,
    tile_inner_lhs: int,
    tile_inner_rhs: int,
    lhs_bytes: int,
    rhs_bytes: int,
    output_bytes: int,
) -> Dict[str, int]:
    lhs_tile_bytes = int(full_outer) * int(tile_inner_lhs) * max(1, int(lhs_bytes))
    rhs_tile_bytes = int(tile_inner_lhs) * int(tile_inner_rhs) * max(1, int(rhs_bytes))
    rhs_tile_bytes += _rhs_stack_overhead_bytes(op_type)
    output_tile_bytes = int(full_outer) * int(tile_inner_rhs) * max(1, int(output_bytes))
    return {
        'lhs_tile_bytes': lhs_tile_bytes,
        'rhs_tile_bytes': rhs_tile_bytes,
        'output_tile_bytes': output_tile_bytes,
        'max_bank_tile_bytes': max(lhs_tile_bytes, rhs_tile_bytes, output_tile_bytes),
        'bank_capacity_bytes': _bank_capacity_bytes(device),
    }


def _supported_microtile_options(op_type: str, generation: str, lhs_dtype, rhs_dtype):
    return get_op_impl_registry().supported_microtilings(
        op_type, generation, (microtile_key(lhs_dtype), microtile_key(rhs_dtype))
    )


def _resolve_tile_cfg(node, device, lhs_dtype, rhs_dtype) -> MatmulMicrotileConfig:
    microtiling_cfg = node.directives.get('microtiling', {}) or {}
    raw = {
        key: int(microtiling_cfg[key]) if key in microtiling_cfg else 0
        for key in ('microtile_m', 'microtile_n', 'microtile_k')
    }
    options = _supported_microtile_options(node.op_type, device.generation, lhs_dtype, rhs_dtype)
    if not options:
        raise ValueError(
            f'{node.name}: no supported tile configs are registered for Generation={device.generation} and '
            f'(input={microtile_key(lhs_dtype)!r}, weight={microtile_key(rhs_dtype)!r}).'
        )

    user_specified = (raw['microtile_m'] > 0) and (raw['microtile_k'] > 0) and (raw['microtile_n'] > 0)
    if user_specified:
        candidate = (raw['microtile_m'], raw['microtile_k'], raw['microtile_n'])
        if candidate not in options:
            raise ValueError(
                f'{node.name}: microtiling {candidate} not supported for Generation={device.generation} and '
                f'(input={microtile_key(lhs_dtype)!r}, weight={microtile_key(rhs_dtype)!r}). Allowed: {options}'
            )
        return MatmulMicrotileConfig(microtile_m=candidate[0], microtile_k=candidate[1], microtile_n=candidate[2])

    default_m, default_k, default_n = options[0]
    return MatmulMicrotileConfig(microtile_m=default_m, microtile_k=default_k, microtile_n=default_n)


def _resolve_numeric(node, device) -> Dict[str, AIEDataType]:
    lhs_tensor = input_tensor_for_role(node, 'lhs')
    rhs_tensor = input_tensor_for_role(node, 'rhs')
    out_tensor = node.outputs[0]
    if lhs_tensor is None or rhs_tensor is None:
        raise ValueError(f'{node.name}: missing lhs/rhs tensor roles for {node.op_type}.')
    if any(t.precision is None for t in (lhs_tensor, rhs_tensor, out_tensor)):
        raise ValueError(f'{node.name}: missing precision metadata for {node.op_type}.')

    resolved = {
        'lhs': resolve_exact_storage_dtype(lhs_tensor.precision, namespace='lhs', layer_name=node.name),
        'rhs': resolve_exact_storage_dtype(rhs_tensor.precision, namespace='rhs', layer_name=node.name),
        'output': resolve_exact_storage_dtype(out_tensor.precision, namespace='output', layer_name=node.name),
    }

    if isinstance(lhs_tensor.precision, FloatIntent):
        if not all(isinstance(t.precision, FloatIntent) for t in (lhs_tensor, rhs_tensor, out_tensor)):
            raise ValueError(f'{node.name}: float {node.op_type} requires lhs/rhs/output to share float precision.')
        resolved['acc'] = AIEDataType(width=32, signed=True, frac=0, c_type='accfloat')
        if node.op_type == 'dense':
            resolved['bias'] = AIEDataType(width=32, signed=True, frac=0, c_type='float')
        return resolved

    lhs_intent = to_quant_intent(lhs_tensor.precision)
    rhs_intent = to_quant_intent(rhs_tensor.precision)

    if int(resolved['lhs'].width) <= 8 and int(resolved['rhs'].width) > 8:
        raise RuntimeError(
            f'{node.name}: unsupported int8 x int16 precision mix for AIE implementations; '
            'no implementation variant available.'
        )

    if node.op_type == 'dense':
        bias_tensor = next((t for t in node.inputs if t.is_parameter and input_role(node, t.name) == 'bias'), None)
        if bias_tensor is not None and bias_tensor.precision is not None:
            bias_intent = to_quant_intent(bias_tensor.precision)
            resolved['bias'] = AIEDataType(
                width=32,
                signed=bool(bias_intent.signed),
                frac=int(lhs_intent.frac + rhs_intent.frac),
                rounding=bias_intent.rounding,
                saturation=bias_intent.saturation,
                c_type='int32_t',
            )
        else:
            resolved['bias'] = AIEDataType(
                width=32, signed=True, frac=int(lhs_intent.frac + rhs_intent.frac), c_type='int32_t'
            )

    acc_tag = infer_accumulator_tag(device, resolved['lhs'], resolved['rhs'], None)
    acc_width = {'acc32': 32, 'acc48': 48, 'acc64': 64}[acc_tag]
    resolved['acc'] = AIEDataType(
        width=acc_width,
        signed=True,
        frac=int(lhs_intent.frac + rhs_intent.frac),
        c_type={32: 'int32_t', 48: 'int64_t', 64: 'int64_t'}[acc_width],
    )
    return resolved


def _parallelism_candidate(
    *,
    op_type: str,
    device,
    in_shape: int,
    out_shape: int,
    lhs_align: int,
    rhs_align: int,
    lhs_bytes: int,
    rhs_bytes: int,
    output_bytes: int,
    full_outer: int,
    cas_num: int,
    cas_length: int,
) -> Optional[MatmulTiling]:
    tile_inner_rhs_raw = (out_shape + cas_num - 1) // cas_num if cas_num else out_shape
    tile_inner_lhs_raw = (in_shape + cas_length - 1) // cas_length if cas_length else in_shape

    if tile_inner_lhs_raw * lhs_bytes % 4 != 0:
        return None

    tile_inner_rhs = align_up(tile_inner_rhs_raw, rhs_align)
    tile_inner_lhs = align_up(tile_inner_lhs_raw, lhs_align)
    bank_usage = _tile_bank_usage(
        op_type=op_type,
        device=device,
        full_outer=full_outer,
        tile_inner_lhs=tile_inner_lhs,
        tile_inner_rhs=tile_inner_rhs,
        lhs_bytes=lhs_bytes,
        rhs_bytes=rhs_bytes,
        output_bytes=output_bytes,
    )
    if int(bank_usage['max_bank_tile_bytes']) > int(bank_usage['bank_capacity_bytes']):
        return None

    return MatmulTiling(
        cas_num=int(cas_num),
        cas_length=int(cas_length),
        tile_inner_lhs_raw=int(tile_inner_lhs_raw),
        tile_inner_lhs=int(tile_inner_lhs),
        tile_inner_rhs_raw=int(tile_inner_rhs_raw),
        tile_inner_rhs=int(tile_inner_rhs),
    )


def _resolve_parallelism(
    node, device, microtiling: MatmulMicrotileConfig, precision: Dict[str, AIEDataType]
) -> MatmulTiling:
    lhs_tensor = input_tensor_for_role(node, 'lhs')
    if lhs_tensor is None:
        raise ValueError(f'{node.name}: missing lhs tensor.')

    lhs_shape = view_shape(node, lhs_tensor, 'inputs')
    in_shape = lhs_shape[-1]
    out_shape = view_shape(node, node.outputs[0], 'outputs')[-1]
    parallel_cfg = node.directives.get('parallelism', {}) or {}
    user_num_chains = parallel_cfg.get('cas_num')
    user_cas_length = parallel_cfg.get('cas_length')
    target_parallel_factor = parallel_cfg.get('parallel_factor')

    lhs_bytes = element_bytes(precision['lhs'])
    rhs_bytes = element_bytes(precision['rhs'])
    output_bytes = element_bytes(precision['output'])

    lhs_align = 2 * microtiling.microtile_k
    rhs_align = 2 * microtiling.microtile_n
    outer_granularity = 2 * microtiling.microtile_m

    last_outer = int(lhs_shape[-2]) if len(lhs_shape) > 1 else 1
    outer_extent = int(math.prod(lhs_shape[:-1]))
    padded_last_outer = align_up(last_outer, outer_granularity)
    full_outer = (outer_extent // max(1, last_outer)) * padded_last_outer

    def _candidate(cas_num, cas_length):
        return _parallelism_candidate(
            op_type=node.op_type,
            device=device,
            in_shape=int(in_shape),
            out_shape=int(out_shape),
            lhs_align=int(lhs_align),
            rhs_align=int(rhs_align),
            lhs_bytes=int(lhs_bytes),
            rhs_bytes=int(rhs_bytes),
            output_bytes=int(output_bytes),
            full_outer=int(full_outer),
            cas_num=int(cas_num),
            cas_length=int(cas_length),
        )

    if user_num_chains and user_cas_length:
        tiling = _candidate(user_num_chains, user_cas_length)
        if tiling is None:
            raise ValueError(f'{node.name}: user-provided parallelism overrides are invalid.')
        return tiling

    max_chain_candidates = min(
        max(1, int(device.rows)),
        max(max(1, int(getattr(device, 'max_mem_out_ports', 0) or 0)), ceildiv(int(out_shape), max(1, rhs_align))),
    )
    max_cas_candidates = min(
        max(1, int(device.columns)),
        max(max(1, int(getattr(device, 'max_mem_in_ports', 0) or 0)), ceildiv(int(in_shape), max(1, lhs_align))),
    )
    chain_candidates = [int(user_num_chains)] if user_num_chains else list(range(1, max_chain_candidates + 1))
    cas_candidates = [int(user_cas_length)] if user_cas_length else list(range(1, max_cas_candidates + 1))

    best: Optional[tuple] = None
    for cas_length in cas_candidates:
        for cas_num in chain_candidates:
            tiling = _candidate(cas_num, cas_length)
            if tiling is None:
                continue

            parallel_factor = tiling.cas_num * tiling.cas_length
            bank_usage = _tile_bank_usage(
                op_type=node.op_type,
                device=device,
                full_outer=int(full_outer),
                tile_inner_lhs=tiling.tile_inner_lhs,
                tile_inner_rhs=tiling.tile_inner_rhs,
                lhs_bytes=int(lhs_bytes),
                rhs_bytes=int(rhs_bytes),
                output_bytes=int(output_bytes),
            )
            utilization_penalty = abs(
                1.0 - float(bank_usage['max_bank_tile_bytes']) / max(1.0, float(bank_usage['bank_capacity_bytes']))
            )
            shape_penalty = max(
                0.0,
                (float(tiling.tile_inner_rhs) - float(tiling.tile_inner_lhs)) / max(1.0, float(tiling.tile_inner_lhs)),
            )
            padding_waste = (
                tiling.tile_inner_lhs * tiling.cas_length
                - int(in_shape)
                + tiling.tile_inner_rhs * tiling.cas_num
                - int(out_shape)
            )
            if target_parallel_factor is not None:
                target_parallel_factor = int(target_parallel_factor)
                score = (
                    int(parallel_factor != target_parallel_factor),
                    abs(parallel_factor - target_parallel_factor),
                    tiling.cas_length,
                    shape_penalty,
                    padding_waste,
                    utilization_penalty,
                )
            else:
                score = (
                    parallel_factor,
                    tiling.cas_length,
                    shape_penalty,
                    padding_waste,
                    utilization_penalty,
                )

            if best is None or score < best[0]:
                best = (score, tiling)

    if best is None:
        raise ValueError(f'{node.name}: no valid parallelism fits tile memory.')
    return best[1]


def _build_io_views(node, microtiling: MatmulMicrotileConfig, tiling: MatmulTiling) -> Dict[str, TensorView]:
    full_inner_lhs = tiling.tile_inner_lhs * tiling.cas_length
    full_inner_out = tiling.tile_inner_rhs * tiling.cas_num
    outer_granularity = 2 * microtiling.microtile_m

    shapes: Dict[str, TensorView] = {}

    for tensor in node.inputs:
        role = input_role(node, tensor.name)
        real = tuple(int(x) for x in view_shape(node, tensor, 'inputs'))

        if role == 'lhs':
            last_outer = int(real[-2]) if len(real) > 1 else 1
            shapes[tensor.name] = build_tensor_view(
                node,
                tensor,
                'inputs',
                full_inner=full_inner_lhs,
                tile_inner=tiling.tile_inner_lhs,
                tile_inner_raw=tiling.tile_inner_lhs_raw,
                full_outer=align_up(last_outer, outer_granularity),
            )
        elif role == 'rhs' and not tensor.is_parameter:
            # matmul activation RHS: outer dim = K slice, inner dim = N slice
            shapes[tensor.name] = build_tensor_view(
                node,
                tensor,
                'inputs',
                full_inner=full_inner_out,
                tile_inner=tiling.tile_inner_rhs,
                tile_inner_raw=tiling.tile_inner_rhs_raw,
                full_outer=full_inner_lhs,
                tile_outer=tiling.tile_inner_lhs,
                tile_outer_raw=tiling.tile_inner_lhs_raw,
            )
        else:
            # parameter (dense weights, bias): no padding applied
            shapes[tensor.name] = build_tensor_view(
                node,
                tensor,
                'inputs',
                full_inner=real[-1] if real else 1,
                tile_inner=real[-1] if real else 1,
                tile_inner_raw=real[-1] if real else 1,
                full_outer=real[-2] if len(real) >= 2 else 1,
            )

    for tensor in node.outputs:
        real = tuple(int(x) for x in view_shape(node, tensor, 'outputs'))
        last_outer = int(real[-2]) if len(real) > 1 else 1
        shapes[tensor.name] = build_tensor_view(
            node,
            tensor,
            'outputs',
            full_inner=full_inner_out,
            tile_inner=tiling.tile_inner_rhs,
            tile_inner_raw=tiling.tile_inner_rhs_raw,
            full_outer=align_up(last_outer, outer_granularity),
        )

    return shapes


@family_resolver('dense')
class DenseResolver:
    op_type = 'dense'

    def resolve(self, node, device, directives=None) -> DenseConfig:
        io_route = dict((directives or {}).get('io_route', {}))
        precision = _resolve_numeric(node, device)
        microtiling = _resolve_tile_cfg(node, device, precision['lhs'], precision['rhs'])
        tiling = _resolve_parallelism(node, device, microtiling, precision)
        io_views = _build_io_views(node, microtiling, tiling)

        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        lhs_perm = io_views[lhs_tensor.name].perm
        is_float = isinstance(lhs_tensor.precision, FloatIntent)

        if is_float:
            shift = 0
        else:
            lhs_frac = to_quant_intent(lhs_tensor.precision).frac
            rhs_frac = to_quant_intent(rhs_tensor.precision).frac
            shift = max(0, int(lhs_frac + rhs_frac - to_quant_intent(node.outputs[0].precision).frac))

        fused_act = node.traits.get('fused_activation')
        use_relu = ((fused_act.data.get('activation') if fused_act else '') or '').lower() == 'relu'

        return DenseConfig(
            precision=precision,
            parallelism=MatmulParallelismConfig(
                cas_length=tiling.cas_length,
                cas_num=tiling.cas_num,
            ),
            microtiling=microtiling,
            io_views=io_views,
            io_route=io_route,
            shift=shift,
            accumulator_tag='accfloat'
            if is_float
            else infer_accumulator_tag(device, precision['lhs'], precision['rhs'], precision.get('acc')),
            rounding_mode='conv_even' if is_float else aie_rounding_token(precision['output']),
            flags=DenseFlags(
                use_relu=use_relu,
                transpose_lhs=bool(lhs_perm is not None and lhs_perm[-1] != (len(lhs_perm) - 1)),
                use_bias=bool(node.metadata.get('use_bias')),
            ),
        )

    def select_variant(self, config: DenseConfig, generation: str):
        return _select_variant(self.op_type, config, generation)


@family_resolver('matmul')
class MatmulResolver:
    op_type = 'matmul'

    def resolve(self, node, device, directives=None) -> MatmulConfig:
        io_route = dict((directives or {}).get('io_route', {}))
        precision = _resolve_numeric(node, device)
        microtiling = _resolve_tile_cfg(node, device, precision['lhs'], precision['rhs'])
        tiling = _resolve_parallelism(node, device, microtiling, precision)
        io_views = _build_io_views(node, microtiling, tiling)

        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        lhs_perm = io_views[lhs_tensor.name].perm
        rhs_perm = io_views[rhs_tensor.name].perm
        is_float = isinstance(rhs_tensor.precision, FloatIntent)

        if is_float:
            shift = 0
        else:
            lhs_frac = to_quant_intent(lhs_tensor.precision).frac
            rhs_frac = to_quant_intent(rhs_tensor.precision).frac
            shift = max(0, int(lhs_frac + rhs_frac - to_quant_intent(node.outputs[0].precision).frac))

        return MatmulConfig(
            precision=precision,
            parallelism=MatmulParallelismConfig(
                cas_length=tiling.cas_length,
                cas_num=tiling.cas_num,
            ),
            microtiling=microtiling,
            io_views=io_views,
            io_route=io_route,
            shift=shift,
            accumulator_tag='accfloat'
            if is_float
            else infer_accumulator_tag(device, precision['lhs'], precision['rhs'], precision.get('acc')),
            rounding_mode='conv_even' if is_float else aie_rounding_token(precision['output']),
            flags=MatmulFlags(
                transpose_lhs=bool(lhs_perm is not None and lhs_perm[-1] != (len(lhs_perm) - 1)),
                transpose_rhs=bool(rhs_perm is not None and rhs_perm[-1] != (len(rhs_perm) - 1)),
            ),
        )

    def select_variant(self, config: MatmulConfig, generation: str):
        return _select_variant(self.op_type, config, generation)
