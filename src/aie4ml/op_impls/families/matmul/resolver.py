from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ....aie_types import AIEDataType, FloatIntent, legality_format
from ....ir import input_role, input_tensor_for_role
from ...family_registry import FamilyResolver, family_resolver
from ...registry import get_op_impl_registry
from ...utils import TensorView, align_up, build_tensor_view, ceildiv
from ...utils.io import view_shape
from ...utils.precision import (
    element_bytes,
    infer_accumulator_tag,
    resolve_exact_storage_dtype,
    to_quant_intent,
)
from .config import MatmulMicrotileConfig


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


_MATMUL_RHS_STACK_BYTES = 1024


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
    rhs_tile_bytes += _MATMUL_RHS_STACK_BYTES if op_type == 'matmul' else 0
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
        op_type, generation, (legality_format(lhs_dtype.format), legality_format(rhs_dtype.format))
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
            f'(input={lhs_dtype.format!r}, weight={rhs_dtype.format!r}).'
        )

    user_specified = (raw['microtile_m'] > 0) and (raw['microtile_k'] > 0) and (raw['microtile_n'] > 0)
    if user_specified:
        candidate = (raw['microtile_m'], raw['microtile_k'], raw['microtile_n'])
        if candidate not in options:
            raise ValueError(
                f'{node.name}: microtiling {candidate} not supported for Generation={device.generation} and '
                f'(input={lhs_dtype.format!r}, weight={rhs_dtype.format!r}). Allowed: {options}'
            )
        return MatmulMicrotileConfig(microtile_m=candidate[0], microtile_k=candidate[1], microtile_n=candidate[2])

    default_m, default_k, default_n = options[0]
    return MatmulMicrotileConfig(microtile_m=default_m, microtile_k=default_k, microtile_n=default_n)


def _resolve_numeric(node, device) -> Dict[str, AIEDataType]:
    lhs_tensor = input_tensor_for_role(node, 'lhs')
    rhs_tensor = input_tensor_for_role(node, 'rhs')
    out_tensor = node.outputs[0]
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
        resolved['acc'] = AIEDataType(format='accfloat', frac=0)
        if node.op_type == 'dense':
            resolved['bias'] = AIEDataType(format='float32', frac=0)
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
                format='int32',
                frac=int(lhs_intent.frac + rhs_intent.frac),
                rounding=bias_intent.rounding,
                saturation=bias_intent.saturation,
            )
        else:
            resolved['bias'] = AIEDataType(format='int32', frac=int(lhs_intent.frac + rhs_intent.frac))

    acc_tag = infer_accumulator_tag(device, resolved['lhs'], resolved['rhs'], None)
    acc_width = {'acc32': 32, 'acc48': 48, 'acc64': 64}[acc_tag]
    resolved['acc'] = AIEDataType(
        format=f'int{acc_width}',
        frac=int(lhs_intent.frac + rhs_intent.frac),
    )
    return resolved


def _resolve_output_scale_shift(node, *, is_float: bool) -> int:
    trait = node.traits.get('output_scale')
    if trait is None:
        return 0
    scale = float(trait.data['scale'])
    if is_float:
        raise NotImplementedError(f'{node.name}: fused output scaling is not implemented for float MatMul-family ops.')
    if scale <= 0.0 or scale > 1.0:
        raise ValueError(f'{node.name}: fused output scale must be in the range (0, 1], got {scale}.')
    shift = int(round(-math.log2(scale)))
    if not math.isclose(scale, math.ldexp(1.0, -shift), rel_tol=0.0, abs_tol=1e-12):
        raise NotImplementedError(
            f'{node.name}: fused output scale {scale} is not a power of two; '
            'fixed-point multiplier scaling is not implemented.'
        )
    return shift


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


def _build_matmul_io_views(node, microtiling: MatmulMicrotileConfig, tiling: MatmulTiling) -> Dict[str, TensorView]:
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


def _validate_matmul_family_rank_contract(node) -> None:
    """Reject batched RHS MatMul until a batched_matmul lowering is explicit.

    The dense/matmul family exposes a 2-D GEMM ABI. Leading LHS/output axes may be compacted
    into M only when RHS is a single broadcastable KxN matrix. A non-broadcast
    RHS leading axis means each compacted M block needs a different KxN tile,
    which must be lowered as parallel rank-2 MatMuls or a batched variant.
    """
    lhs_tensor = input_tensor_for_role(node, 'lhs')
    rhs_tensor = input_tensor_for_role(node, 'rhs')

    lhs_shape = tuple(int(x) for x in view_shape(node, lhs_tensor, 'inputs'))
    rhs_shape = tuple(int(x) for x in view_shape(node, rhs_tensor, 'inputs'))
    if len(lhs_shape) < 2:
        raise ValueError(f'{node.name}: {node.op_type}.v1 requires rank>=2 LHS tensors, got {lhs_shape}.')
    if len(rhs_shape) < 2:
        raise ValueError(f'{node.name}: {node.op_type}.v1 requires rank>=2 RHS tensors, got {rhs_shape}.')
    rhs_batch = rhs_shape[:-2] if len(rhs_shape) > 2 else ()
    if rhs_batch and any(int(dim) != 1 for dim in rhs_batch):
        raise ValueError(
            f'{node.name}: matmul.v1 does not support batched RHS MatMul with non-broadcast leading axes; '
            f'lhs shape {lhs_shape}, rhs shape {rhs_shape}. Lower to parallel rank-2 MatMul ops or add a '
            'batched_matmul variant.'
        )


@family_resolver('dense')
class DenseFamilyResolver(FamilyResolver):
    op_type = 'dense'

    def validate_structure(self, node, _device) -> None:
        _validate_matmul_family_rank_contract(node)


@family_resolver('matmul')
class MatmulFamilyResolver(FamilyResolver):
    op_type = 'matmul'

    def validate_structure(self, node, _device) -> None:
        _validate_matmul_family_rank_contract(node)
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        if rhs_tensor.is_parameter:
            raise ValueError(
                f'{node.name}: matmul.v1 requires a runtime RHS tensor. Constant RHS MatMul must lower to dense, '
                'parallel rank-2 MatMul ops, or a static/batched matmul variant.'
            )
