# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Policy-driven resolver registry for AIE attribute resolution."""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..aie_types import AIEDataType, QuantIntent, RoundingMode, SaturationMode
from ..ir import ResolvedAttributes, TraitInstance
from ..kernel_registry import get_kernel_registry

log = logging.getLogger(__name__)


ResolverFn = Callable[['LayerResolveContext'], None]


@dataclass(frozen=True)
class LayerPolicy:
    """Describes the resolver pipeline for a layer class."""

    namespaces: Tuple[str, ...]
    resolvers: Tuple[ResolverFn, ...]
    requires_numeric: bool = False


@dataclass
class LayerResolveContext:
    """State shared across resolver functions."""

    model: Any
    backend_ctx: Any
    node: Any
    layer_name: str
    layer_class: str
    policy: LayerPolicy
    config: Dict[str, Any]
    quant: Dict[str, Any]
    device: Any
    global_config: Dict[str, Any]
    attributes: ResolvedAttributes
    state: Dict[str, Any] = field(default_factory=dict)

    def numeric(self) -> Optional['NumericBundle']:
        return self.state.get('numeric')

    def set_numeric(self, numeric: 'NumericBundle') -> None:
        self.state['numeric'] = numeric

    def set_parallelism(self, parallelism: 'ParallelismResult') -> None:
        self.state['parallelism'] = parallelism

    def parallelism(self) -> Optional['ParallelismResult']:
        return self.state.get('parallelism')


@dataclass
class NumericBundle:
    """Collection of resolved numeric precisions."""

    dtypes: Dict[str, AIEDataType]

    def get(self, key: str) -> Optional[AIEDataType]:
        return self.dtypes.get(key)

    def items(self):
        return self.dtypes.items()

    def to_attribute_map(self) -> Dict[str, AIEDataType]:
        filtered: Dict[str, AIEDataType] = {}
        for key, dtype in self.dtypes.items():
            if dtype is None:
                continue
            width = int(getattr(dtype, 'width', 0) or 0)
            if width > 0:
                filtered[key] = dtype
        return filtered


@dataclass
class ParallelismResult:
    """Outcome of the parallelism resolver."""

    cas_num: int = 1
    cas_length: int = 1
    input_slice: int = 0
    output_slice: int = 0
    input_slice_raw: int = 0
    output_slice_raw: int = 0
    weight_tile_bytes: int = 0
    parallel_factor: int = 1
    input_alignment: int = 1
    output_alignment: int = 1
    padded_independent_extent: int = 1
    independent_extent: int = 1
    padded_in_features: int = 0
    padded_out_features: int = 0


M_GRANULARITY = 2
K_GRANULARITY = 2
N_GRANULARITY = 2

KERNEL_REGISTRY = get_kernel_registry()

ACC_TAG_WIDTHS = {
    'acc32': 32,
    'acc48': 48,
    'acc64': 64,
}

ROUNDING_TOKEN_MAP: Dict[RoundingMode, str] = {
    RoundingMode.TRN: 'floor',
    RoundingMode.RND_MIN_INF: 'floor',
    RoundingMode.RND_INF: 'ceil',
    RoundingMode.RND: 'symmetric_inf',
    RoundingMode.TRN_ZERO: 'symmetric_zero',
    RoundingMode.RND_ZERO: 'symmetric_zero',
    RoundingMode.RND_CONV: 'conv_even',
}


def _normalize_precision_name(name: str) -> str:
    for suffix in ('_precision', '_dtype'):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _ctype_for_width(width: int, signed: bool) -> str:
    width = max(1, int(width))
    if signed:
        if width <= 8:
            return 'int8_t'
        if width <= 16:
            return 'int16_t'
        if width <= 32:
            return 'int32_t'
        return 'int64_t'
    if width <= 8:
        return 'uint8_t'
    if width <= 16:
        return 'uint16_t'
    if width <= 32:
        return 'uint32_t'
    return 'uint64_t'


def _to_quant_intent(precision: Any) -> QuantIntent:
    if isinstance(precision, QuantIntent):
        return precision
    if isinstance(precision, AIEDataType):
        return QuantIntent(
            width=int(precision.width),
            frac=int(precision.frac),
            signed=bool(precision.signed),
            rounding=precision.rounding,
            saturation=precision.saturation,
        )
    raise TypeError(f'Unsupported precision representation {type(precision)}')


def _resolve_storage_width(width: int, *, allowed: Tuple[int, ...], namespace: str, layer_name: str) -> int:
    width = int(width)
    if width <= 0:
        raise ValueError(f'{layer_name}: invalid {namespace} width {width}')
    for candidate in allowed:
        if width <= candidate:
            return candidate
    raise ValueError(f'{layer_name}: {namespace} width {width} exceeds supported widths {allowed}')


def _resolve_storage_dtype(
    intent: QuantIntent, *, allowed: Tuple[int, ...], namespace: str, layer_name: str
) -> AIEDataType:
    storage_width = _resolve_storage_width(intent.width, allowed=allowed, namespace=namespace, layer_name=layer_name)
    return AIEDataType(
        width=storage_width,
        signed=bool(intent.signed),
        frac=int(intent.frac),
        rounding=intent.rounding,
        saturation=intent.saturation,
        c_type=_ctype_for_width(storage_width, bool(intent.signed)),
    )


def resolve_numeric(ctx: LayerResolveContext) -> None:
    """Resolve backend storage types and post-shift from quantization intent."""

    precision_entries: Dict[str, Any] = {}
    for key, value in ctx.quant.items():
        if key.endswith('_precision'):
            if value is None:
                raise RuntimeError(f'{ctx.layer_name}: quant metadata "{key}" is None')
            precision_entries[key] = value

    allowed = {'input', 'output'}
    if ctx.policy.requires_numeric:
        allowed |= {'weight', 'bias', 'acc'}

    required: List[str] = []
    if ctx.policy.requires_numeric:
        required.extend(['input', 'output', 'weight'])
        if ctx.node.metadata['use_bias']:
            required.append('bias')

    missing = []
    for name in required:
        k = f'{name}_precision'
        if k not in precision_entries:
            missing.append(k)
    if missing:
        raise RuntimeError(f'{ctx.layer_name}: missing quant intent {", ".join(sorted(missing))}')

    intents: Dict[str, QuantIntent] = {}
    for precision_key, precision in precision_entries.items():
        alias = _normalize_precision_name(precision_key)
        if alias not in allowed:
            continue
        intents[alias] = _to_quant_intent(precision)

    resolved: Dict[str, AIEDataType] = {}

    if 'input' in intents:
        resolved['input'] = _resolve_storage_dtype(
            intents['input'],
            allowed=(4, 8, 16, 32),
            namespace='input',
            layer_name=ctx.layer_name,
        )
    if 'output' in intents:
        resolved['output'] = _resolve_storage_dtype(
            intents['output'],
            allowed=(4, 8, 16, 32),
            namespace='output',
            layer_name=ctx.layer_name,
        )

    if ctx.policy.requires_numeric:
        resolved['weight'] = _resolve_storage_dtype(
            intents['weight'],
            allowed=(4, 8, 16, 32),
            namespace='weight',
            layer_name=ctx.layer_name,
        )

        if 'bias' in intents:
            # NOTE: keep bias storage fixed to 32 bits for now;
            bias_width = 32
            resolved['bias'] = AIEDataType(
                width=bias_width,
                signed=bool(intents['bias'].signed),
                # Bias is consumed in accumulator scale by the AIE dense kernels.
                frac=int(intents['input'].frac + intents['weight'].frac),
                rounding=intents['bias'].rounding,
                saturation=intents['bias'].saturation,
                c_type=_ctype_for_width(bias_width, bool(intents['bias'].signed)),
            )
        else:
            bias_width = 32
            accum_frac = int(intents['input'].frac + intents['weight'].frac)
            resolved['bias'] = AIEDataType(
                width=bias_width,
                signed=True,
                frac=accum_frac,
                rounding=RoundingMode.TRN,
                saturation=SaturationMode.SAT,
                c_type=_ctype_for_width(bias_width, True),
            )

        in_width = int(resolved['input'].width)
        w_width = int(resolved['weight'].width)
        if in_width <= 8 and w_width > 8:
            raise RuntimeError(
                f'{ctx.layer_name}: unsupported int8 x int16 precision mix for AIE kernels; '
                'no kernel variant available.'
            )

        shift = int(intents['input'].frac + intents['weight'].frac - intents['output'].frac)
        if shift < 0:
            log.warning(
                'Layer %s: computed shift=%d (requires left-shift) but negative shifts are unsafe on AIE-ML/XDNA; '
                'forcing shift=0. Bit-exactness will be lost. Consider increasing output fractional bits or '
                'reducing accumulator fractional depth so output_frac ≤ accum_frac.',
                ctx.layer_name,
                shift,
            )
            shift = 0
        ctx.attributes.scalars['shift'] = shift

        acc_tag = _infer_accumulator_tag(ctx.device, resolved['input'], resolved['weight'], None)
        ctx.attributes.scalars['accumulator_tag'] = acc_tag
        ctx.attributes.scalars['rounding_mode'] = _aie_rounding_token(resolved['output'])

        if 'acc' not in resolved:
            acc_width = ACC_TAG_WIDTHS[acc_tag]
            resolved['acc'] = AIEDataType(
                width=acc_width,
                signed=True,
                frac=int(intents['input'].frac + intents['weight'].frac),
                rounding=RoundingMode.TRN,
                saturation=SaturationMode.SAT,
                c_type=_ctype_for_width(acc_width, True),
            )

    numeric = NumericBundle(resolved)
    ctx.attributes.numeric.update(numeric.to_attribute_map())
    ctx.set_numeric(numeric)


def _supported_tile_options(gen: str, in_w: int, w_w: int):
    return KERNEL_REGISTRY.supported_tilings('dense', gen, int(in_w), int(w_w))


def _extract_tile_cfg(user_cfg: Dict[str, Any]) -> Dict[str, int]:
    tile_cfg = user_cfg['tiling'] if 'tiling' in user_cfg else {}
    out: Dict[str, int] = {}
    for key in ('tile_m', 'tile_n', 'tile_k'):
        if key in user_cfg:
            out[key] = int(user_cfg[key])
        elif key in tile_cfg:
            out[key] = int(tile_cfg[key])
        else:
            out[key] = 0
    return out


def _resolve_tile_cfg(
    layer_name: str,
    user_cfg: Dict[str, Any],
    device: Any,
    input_dtype: Optional[AIEDataType],
    weight_dtype: Optional[AIEDataType],
) -> Dict[str, int]:
    raw = _extract_tile_cfg(user_cfg)
    input_width = int(getattr(input_dtype, 'width', 0) or 0)
    weight_width = int(getattr(weight_dtype, 'width', 0) or 0)
    generation = getattr(device, 'generation', '') or ''

    options = _supported_tile_options(generation, input_width, weight_width)
    if not options:
        raise ValueError(
            f'{layer_name}: no supported tile configs are registered for Generation={generation} and '
            f'(input={input_width}b, weight={weight_width}b); cannot validate user tiling.'
        )

    user_specified = (raw['tile_m'] > 0) and (raw['tile_k'] > 0) and (raw['tile_n'] > 0)
    if user_specified:
        candidate = (raw['tile_m'], raw['tile_k'], raw['tile_n'])
        if candidate not in options:
            raise ValueError(
                f'{layer_name}: tiling {candidate} not supported for Generation={generation} and '
                f'(input={input_width}b, weight={weight_width}b). Allowed: {options}'
            )
        return {'tile_m': candidate[0], 'tile_k': candidate[1], 'tile_n': candidate[2]}

    default_m, default_k, default_n = options[0]
    return {'tile_m': default_m, 'tile_k': default_k, 'tile_n': default_n}


def resolve_tiling(ctx: LayerResolveContext) -> None:
    numeric = ctx.numeric()
    if numeric is None:
        return
    input_dtype = numeric.get('input')
    weight_dtype = numeric.get('weight')
    if input_dtype is None or weight_dtype is None:
        return

    tile_cfg = _resolve_tile_cfg(ctx.layer_name, ctx.config, ctx.device, input_dtype, weight_dtype)
    ctx.attributes.tiling.update(tile_cfg)
    ctx.state['tile_cfg'] = tile_cfg


def _device_lane_bytes(device: Any) -> int:
    norm = (getattr(device, 'generation', '') or '').upper()
    if any(token in norm for token in ('AIE-ML', 'AIE-MLV2', 'MLV2', 'XDNA', 'AIE2')):
        return 16  # NOTE this should be 32 in practice but compiler doesn't complain
    return 16


def _element_bytes(dtype: Optional[AIEDataType]) -> int:
    """Return number of bytes needed to store one element of this dtype."""
    if not dtype or not getattr(dtype, 'width', None):
        return 1

    width = int(dtype.width)
    return max(1, math.ceil(width / 8))


def _features_from_bytes(byte_alignment: int, element_bytes: int) -> int:
    if byte_alignment <= 0:
        return 1
    element_bytes = max(1, element_bytes)
    return max(1, math.ceil(byte_alignment / element_bytes))


def _lcm(a: int, b: int) -> int:
    if a <= 0:
        return max(1, b)
    if b <= 0:
        return max(1, a)
    return abs(a * b) // math.gcd(a, b)


def _lcm_many(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result = _lcm(result, int(value))
    return result


def _input_slice_alignment(device: Any, tile_k: int, element_bytes: int) -> int:
    base = max(1, K_GRANULARITY * max(1, tile_k))
    lane = _features_from_bytes(_device_lane_bytes(device), element_bytes)
    plio = _features_from_bytes(4, element_bytes)
    return _lcm_many([base, lane, plio])


def _output_slice_alignment(device: Any, tile_n: int, element_bytes: int) -> int:
    base = max(1, N_GRANULARITY * max(1, tile_n))
    lane = _features_from_bytes(_device_lane_bytes(device), element_bytes)
    plio = _features_from_bytes(4, element_bytes)
    return _lcm_many([base, lane, plio])


def _align_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        return max(0, value)
    return ((int(value) + multiple - 1) // multiple) * multiple


def _tensor_feature_dim(tensor, layer_name: str, role: str) -> int:
    if tensor is None:
        raise ValueError(f'{layer_name}: missing tensor for {role}.')
    shape = getattr(tensor, 'shape', None)
    if not shape:
        raise ValueError(f'{layer_name}: tensor shape missing for {role}.')
    dim = int(shape[-1])
    if dim <= 0:
        raise ValueError(f'{layer_name}: invalid {role} features {dim}.')
    return dim


def _aligned_batch_size(batch: int, tile_m: int) -> int:
    two_m = max(1, 2 * max(1, tile_m))
    return _align_up(int(batch), two_m)


def _independent_extent(ctx: LayerResolveContext) -> Tuple[int, int]:
    logical = [int(x) for x in ctx.node.inputs[0].shape]
    if len(logical) < 2:
        raise ValueError(f'{ctx.layer_name}: tensor rank must be >=2, got {len(logical)}.')
    independent = logical[:-1]
    extent = int(math.prod(independent))
    last_indep = int(independent[-1])
    return extent, last_indep


def _pad_logical_shape(
    logical: List[int],
    feature_axis: int,
    independent_axes: List[int],
    padded_feat: int,
    tile_m: int,
) -> List[int]:
    padded = list(logical)
    padded[feature_axis] = int(padded_feat)
    if independent_axes:
        last_axis = int(independent_axes[-1])
        padded[last_axis] = _align_up(int(padded[last_axis]), 2 * max(1, int(tile_m)))
    return padded


def _resolve_io_shapes(ctx: LayerResolveContext) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
    io_trait = ctx.node.traits.get('io_view')
    views = io_trait.data if io_trait else {'inputs': {}, 'outputs': {}}
    tile_m = ctx.attributes.tiling['tile_m']
    shapes: Dict[str, Dict[str, Dict[str, List[int]]]] = {'inputs': {}, 'outputs': {}}

    for t in ctx.node.inputs:
        view = views['inputs'][t.name]
        logical = [int(x) for x in t.shape]
        real = list(logical)
        padded = _pad_logical_shape(
            logical,
            view['feature_axis'],
            view['independent_axes'],
            padded_feat=int(ctx.attributes.scalars['padded_in_features']),
            tile_m=tile_m,
        )
        shapes['inputs'][t.name] = {'logical': logical, 'real': real, 'padded': padded}

    for t in ctx.node.outputs:
        view = views['outputs'][t.name]
        logical = [int(x) for x in t.shape]
        real = list(logical)
        padded = _pad_logical_shape(
            logical,
            view['feature_axis'],
            view['independent_axes'],
            padded_feat=int(ctx.attributes.scalars['padded_out_features']),
            tile_m=tile_m,
        )
        shapes['outputs'][t.name] = {'logical': logical, 'real': real, 'padded': padded}

    return shapes


def _aligned_input_features(
    in_feat: int,
    cas_length: int,
    tile_k: int,
    device: Any,
    element_bytes: int,
) -> int:
    slice_alignment = _input_slice_alignment(device, tile_k, element_bytes)
    block = max(1, int(cas_length) * slice_alignment)
    return _align_up(int(in_feat), block)


def _validate_parallel_override(
    layer_name: str,
    chains: int,
    cas: int,
    n_in: int,
    n_out: int,
    align_k: int,
    align_n: int,
    weight_bytes: int,
    input_elem_bytes: int,
    device: Any,
    allow_failure: bool = False,
) -> Optional[Dict[str, Any]]:
    if chains <= 0 or cas <= 0:
        raise ValueError(f'{layer_name}: cas_num and cas_length must be positive.')

    out_slice_raw = (n_out + chains - 1) // chains if chains else n_out
    in_slice_raw = (n_in + cas - 1) // cas if cas else n_in

    if in_slice_raw * input_elem_bytes % 4 != 0:  # PLIO 32-bit align
        if allow_failure:
            return None
        raise ValueError(f'{layer_name}: raw IN slice not 32-bit aligned ({in_slice_raw * input_elem_bytes}B).')

    out_slice = _align_up(out_slice_raw, align_n)
    in_slice = _align_up(in_slice_raw, align_k)

    tile_bytes = in_slice * out_slice * max(1, weight_bytes)
    per_tile_limit = max(1, int(getattr(device, 'weight_mem_bytes', 0) or 1))

    if tile_bytes > per_tile_limit:
        if allow_failure:
            return None
        raise ValueError(
            f'{layer_name}: no valid (cas_num, cas_length) fits tile memory '
            f'(requested {chains}x{cas}, needs {tile_bytes}B > {per_tile_limit}B).'
        )

    return {
        'cas_num': chains,
        'cas_length': cas,
        'input_slice_raw': in_slice_raw,
        'output_slice_raw': out_slice_raw,
        'input_slice': in_slice,
        'output_slice': out_slice,
        'tile_bytes': tile_bytes,
        'balance': abs(in_slice - out_slice),
    }


def _resolve_parallelism_numeric(
    ctx: LayerResolveContext,
    numeric: NumericBundle,
    tile_cfg: Dict[str, int],
) -> ParallelismResult:
    layer_name = ctx.layer_name
    if not ctx.node.inputs or not ctx.node.outputs:
        raise ValueError(f'{layer_name}: node is missing input or output tensors.')

    in_shape = _tensor_feature_dim(ctx.node.inputs[0], layer_name, 'input')
    out_shape = _tensor_feature_dim(ctx.node.outputs[0], layer_name, 'output')

    parallel_cfg = ctx.config.get('parallelism', {}) or {}
    user_num_chains = ctx.config.get('cas_num', parallel_cfg.get('cas_num'))
    user_cas_length = ctx.config.get('cas_length', parallel_cfg.get('cas_length'))
    user_target = parallel_cfg.get('parallel_factor')
    target_parallel_factor = None if user_target in (None, 0, '') else int(user_target)

    def _validate_positive(name: str, value: Any) -> None:
        if value is None:
            return
        ivalue = int(value)
        if ivalue <= 0:
            raise ValueError(f'{layer_name}: {name} must be positive, got {value!r}.')

    _validate_positive('cas_num', user_num_chains)
    _validate_positive('cas_length', user_cas_length)
    _validate_positive('target_parallel_factor', target_parallel_factor)

    tile_m = int(tile_cfg['tile_m'])
    tile_n = int(tile_cfg['tile_n'])
    tile_k = int(tile_cfg['tile_k'])
    if tile_m <= 0 or tile_n <= 0 or tile_k <= 0:
        raise ValueError(f'{layer_name}: tiling not resolved before parallelism.')

    input_bytes = _element_bytes(numeric.get('input'))
    output_bytes = _element_bytes(numeric.get('output'))
    weight_dtype = numeric.get('weight')
    weight_width = int(getattr(weight_dtype, 'width'))
    weight_bytes = max(1, math.ceil(weight_width / 8))

    align_k = _input_slice_alignment(ctx.device, tile_k, input_bytes)
    align_n = _output_slice_alignment(ctx.device, tile_n, output_bytes)

    max_out_ports = max(1, int(getattr(ctx.device, 'max_mem_out_ports', 0) or 0))
    max_in_ports = max(1, int(getattr(ctx.device, 'max_mem_in_ports', 0) or 0))

    if user_num_chains and int(user_num_chains) > max_out_ports:
        log.warning(
            '%s: cas_num override %s exceeds single memtile out-ports %s; '
            'MemoryPlan will shard across multiple memtiles.',
            layer_name,
            user_num_chains,
            max_out_ports,
        )
    if user_cas_length and int(user_cas_length) > max_in_ports:
        log.warning(
            '%s: cas_length override %s exceeds single memtile in-ports %s; '
            'MemoryPlan will shard across multiple memtiles.',
            layer_name,
            user_cas_length,
            max_in_ports,
        )

    if user_num_chains and user_cas_length:
        override = _validate_parallel_override(
            layer_name,
            int(user_num_chains),
            int(user_cas_length),
            in_shape,
            out_shape,
            align_k,
            align_n,
            weight_bytes,
            _element_bytes(numeric.get('input')),
            ctx.device,
        )
        if override is None:
            raise ValueError(f'{layer_name}: user-provided parallelism overrides are invalid.')
        parallel_factor = int(user_num_chains) * int(user_cas_length)
        candidate = {
            **override,
            'cas_num': int(user_num_chains),
            'cas_length': int(user_cas_length),
            'parallel_factor': parallel_factor,
        }
    else:
        chain_candidates = [int(user_num_chains)] if user_num_chains else list(range(1, max_out_ports + 1))
        cas_candidates = [int(user_cas_length)] if user_cas_length else list(range(1, max_in_ports + 1))

        best_pair = None  # (score_tuple, cand_dict)
        for cas in cas_candidates:
            for chains in chain_candidates:
                cand = _validate_parallel_override(
                    layer_name,
                    chains,
                    cas,
                    in_shape,
                    out_shape,
                    align_k,
                    align_n,
                    weight_bytes,
                    _element_bytes(numeric.get('input')),
                    ctx.device,
                    allow_failure=True,
                )
                if cand is None:
                    continue

                parallel_factor = chains * cas
                # --- scoring ---
                per_tile_limit = max(1, int(getattr(ctx.device, 'weight_mem_bytes', 0)))
                utilization_penalty = abs(
                    1.0 - (cand['tile_bytes'] / per_tile_limit)
                )  # prefer to use the full tile memory
                # aspect = cand['input_slice'] / cand['output_slice']
                shape_penalty = max(
                    0.0, (cand['output_slice'] - cand['input_slice']) / max(1.0, cand['input_slice'])
                )  # penalize OUT >> IN
                padding_waste = (cand['input_slice'] * cas - in_shape) + (
                    cand['output_slice'] * chains - out_shape
                )  # Penalize alignment padding
                match_penalty = 0 if target_parallel_factor is None else abs(parallel_factor - target_parallel_factor)
                if target_parallel_factor is not None:
                    exact_miss = int(parallel_factor != target_parallel_factor)  # 0 for exact, 1 otherwise
                    match_penalty = abs(parallel_factor - target_parallel_factor)
                    score = (
                        exact_miss,
                        match_penalty,
                        utilization_penalty,
                        shape_penalty,
                        padding_waste,
                        -parallel_factor,
                    )
                else:
                    score = (utilization_penalty, shape_penalty, padding_waste, -parallel_factor)

                if best_pair is None or score < best_pair[0]:
                    best_pair = (
                        score,
                        {
                            **cand,
                            'cas_num': chains,
                            'cas_length': cas,
                            'parallel_factor': parallel_factor,
                        },
                    )

        if best_pair is None:
            raise ValueError(
                f"{layer_name}: no valid (cas_num, cas_length) fits tile memory "
                f"(n_in={in_shape}, n_out={out_shape}, tile limit={getattr(ctx.device, 'weight_mem_bytes', 0)}B). "
                "Try adjusting: parallelism, tiling, weight bitwidth, or device memory."
            )
        candidate = best_pair[1]

    cas_num = int(candidate['cas_num'])
    cas_length = int(candidate['cas_length'])
    raw_in_slice = int(candidate['input_slice_raw'])
    raw_out_slice = int(candidate['output_slice_raw'])
    in_slice = int(candidate['input_slice'])
    out_slice = int(candidate['output_slice'])

    # (Already aligned by validator; these are no-ops but safe)
    if in_slice > 0:
        in_slice = _align_up(in_slice, align_k)
    if out_slice > 0:
        out_slice = _align_up(out_slice, align_n)

    indep_extent, last_indep = _independent_extent(ctx)
    padded_last = _aligned_batch_size(last_indep, tile_m)
    padded_indep = (indep_extent // max(1, last_indep)) * padded_last
    padded_in = _aligned_input_features(in_shape, cas_length, tile_k, ctx.device, input_bytes)
    padded_in = max(padded_in, in_slice * max(1, cas_length))
    padded_out = out_slice * max(1, cas_num)

    return ParallelismResult(
        cas_num=cas_num,
        cas_length=cas_length,
        input_slice=in_slice,
        output_slice=out_slice,
        input_slice_raw=raw_in_slice,
        output_slice_raw=raw_out_slice,
        weight_tile_bytes=int(candidate['tile_bytes']),
        parallel_factor=int(candidate['parallel_factor']),
        input_alignment=align_k,
        output_alignment=align_n,
        padded_independent_extent=int(padded_indep),
        independent_extent=int(indep_extent),
        padded_in_features=int(padded_in),
        padded_out_features=int(padded_out),
    )


def resolve_parallelism(ctx: LayerResolveContext) -> None:
    numeric = ctx.numeric()
    if numeric is None:
        raise RuntimeError(f'{ctx.layer_name}: numeric precisions missing before parallelism resolution.')
    tile_cfg = ctx.state['tile_cfg']
    result = _resolve_parallelism_numeric(ctx, numeric, tile_cfg)

    ctx.set_parallelism(result)
    ctx.attributes.parallelism.update(
        {
            'parallel_factor': int(result.parallel_factor),
            'cas_num': int(result.cas_num),
            'cas_length': int(result.cas_length),
            'weight_tile_bytes': int(result.weight_tile_bytes),
            'input_alignment': int(result.input_alignment),
            'output_alignment': int(result.output_alignment),
        }
    )
    ctx.attributes.slices.update(
        {
            'input': int(result.input_slice),
            'input_raw': int(result.input_slice_raw),
            'output': int(result.output_slice),
            'output_raw': int(result.output_slice_raw),
        }
    )


def resolve_flags(ctx: LayerResolveContext) -> None:
    flags_cfg = ctx.config.get('flags', {}) or {}
    if isinstance(flags_cfg, dict):
        ctx.attributes.flags.update(flags_cfg)

    fused_trait = ctx.node.traits.get('fused_activation')
    activation = (fused_trait.data.get('activation') if fused_trait else '') or ''
    ctx.attributes.flags['use_relu'] = activation.lower() == 'relu'


def resolve_io_route(ctx: LayerResolveContext) -> None:
    """
    Normalize and default IO routing:
      io_route.inputs.<tensor>  = "direct" | "memtile" | "plio"
      io_route.outputs.<tensor> = "direct" | "memtile" | "plio"

    Meaning:
      - direct:  attempt direct kernel<->kernel transport (only if legal; else memtile fallback in plan pass)
      - memtile: force shared_buffer transport
      - plio:    export/import through extra top_graph ports (intermediate debug IO)
    """
    r = ctx.attributes.io_route
    r.setdefault('inputs', {})
    r.setdefault('outputs', {})

    for t in ctx.node.inputs:
        r['inputs'].setdefault(t.name, 'direct')

    for t in ctx.node.outputs:
        r['outputs'].setdefault(t.name, 'direct')

    user = ctx.config.get('io_route', {})
    for d in ('inputs', 'outputs'):
        if isinstance(user.get(d), dict):
            r[d].update(user[d])


def resolve_io_view(ctx: LayerResolveContext) -> None:
    io_trait = ctx.node.traits.get('io_view')
    if io_trait is None:
        io_trait = TraitInstance('io_view', {'inputs': {}, 'outputs': {}})
        ctx.node.add_trait(io_trait)
    data = io_trait.data
    data.setdefault('inputs', {})
    data.setdefault('outputs', {})

    gen = (getattr(ctx.device, 'generation', '') or '').upper()
    max_rank = 5 if 'AIE-MLV2' in gen else 4

    def _default_view(rank: int) -> Dict[str, Any]:
        return {
            'layout': 'channels_last',
            'feature_axis': rank - 1,
            'independent_axes': list(range(rank - 1)),
            'buffer_order': list(reversed(range(rank))),
        }

    for t in ctx.node.inputs:
        logical = [int(x) for x in t.shape]
        rank = len(logical)
        if rank > max_rank:
            raise ValueError(
                f'{ctx.layer_name}: tensor rank {rank} exceeds max {max_rank} for {ctx.device.generation}.'
            )
        if t.name not in data['inputs']:
            data['inputs'][t.name] = _default_view(rank)

    for t in ctx.node.outputs:
        logical = [int(x) for x in t.shape]
        rank = len(logical)
        if rank > max_rank:
            raise ValueError(
                f'{ctx.layer_name}: tensor rank {rank} exceeds max {max_rank} for {ctx.device.generation}.'
            )
        if t.name not in data['outputs']:
            data['outputs'][t.name] = _default_view(rank)


def resolve_staging(ctx: LayerResolveContext) -> None:
    """
    Attach user-provided staging overrides.
      staging.inputs.<tensor>  = { scheme: ... }
      staging.outputs.<tensor> = { scheme: ... }
    """

    st = ctx.attributes.staging
    st['inputs'] = {}
    st['outputs'] = {}

    user = ctx.config.get('staging')
    if not isinstance(user, dict):
        return

    for direction in ('inputs', 'outputs'):
        if direction not in user:
            continue
        overrides = user[direction]
        if not isinstance(overrides, dict):
            raise ValueError(f'{ctx.layer_name}: staging.{direction} must be a dict.')

        for tensor_name, value in overrides.items():
            if not isinstance(value, dict):
                raise ValueError(f'{ctx.layer_name}: staging.{direction}.{tensor_name} must be a dict.')
            st[direction][tensor_name] = dict(value)


def _stable_pack_key(layer_name: str, resolved: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(layer_name.encode('utf-8'))
    digest.update(b'|AIE|')
    normalized: List[Tuple[str, Any]] = []
    for key in sorted(resolved.keys()):
        value = resolved[key]
        if isinstance(value, dict):
            normalized.append((key, tuple(sorted(value.items()))))
        elif isinstance(value, list):
            normalized.append((key, tuple(value)))
        else:
            normalized.append((key, value))
    digest.update(repr(tuple(normalized)).encode('utf-8'))
    return digest.hexdigest()


def resolve_pack(ctx: LayerResolveContext) -> None:
    if not ctx.attributes.tiling or not ctx.attributes.parallelism:
        return

    numeric = ctx.attributes.numeric
    pack_input = {
        'tiling': {k: ctx.attributes.tiling[k] for k in ('tile_k', 'tile_n', 'tile_m') if k in ctx.attributes.tiling},
        'parallelism': {
            'cas_num': ctx.attributes.parallelism.get('cas_num'),
            'cas_length': ctx.attributes.parallelism.get('cas_length'),
        },
        'element': {
            'weight_width': getattr(numeric.get('weight'), 'width', None),
            'weight_signed': getattr(numeric.get('weight'), 'signed', None),
        },
    }
    ctx.attributes.pack['key'] = _stable_pack_key(ctx.layer_name, pack_input)


def resolve_placement(ctx: LayerResolveContext) -> None:
    placement_cfg = ctx.config.get('placement')
    if not isinstance(placement_cfg, dict) or not placement_cfg:
        return
    placement: Dict[str, int] = {}
    if 'col' in placement_cfg:
        placement['col'] = int(placement_cfg['col'])
    if 'row' in placement_cfg:
        placement['row'] = int(placement_cfg['row'])
    ctx.attributes.placement = placement if placement else None


def _acc_tag_from_width(width: int) -> Optional[str]:
    for tag, bits in ACC_TAG_WIDTHS.items():
        if bits == width:
            return tag
    return None


def _infer_accumulator_tag(
    device: Any,
    input_dtype: Optional[AIEDataType],
    weight_dtype: Optional[AIEDataType],
    acc_precision: Optional[AIEDataType],
) -> Optional[str]:
    if acc_precision is not None and acc_precision.width:
        tag = _acc_tag_from_width(int(acc_precision.width))
        if tag is None:
            raise ValueError(
                f'Unsupported accumulator precision width {acc_precision.width}; ' 'expected one of 32, 48 or 64 bits.'
            )
        return tag

    if input_dtype is None or weight_dtype is None:
        return None

    in_w = int(getattr(input_dtype, 'width', 0) or 0)
    w_w = int(getattr(weight_dtype, 'width', 0) or 0)
    norm_gen = (getattr(device, 'generation', '') or '').upper()
    is_ml = norm_gen.startswith('AIE-ML') or 'XDNA' in norm_gen

    if not is_ml:
        if in_w <= 8 and w_w <= 8:
            return 'acc32'
        if in_w <= 16 and w_w <= 16:
            return 'acc48'
        raise ValueError(
            f'No accumulator tag registered for AIE generation "{device.generation}" with '
            f'input {in_w}-bit and weight {w_w}-bit precisions.'
        )

    if max(in_w, w_w) <= 8:
        return 'acc32'
    if {in_w, w_w} in ({8, 16}, {16, 8}):
        return 'acc32'
    if max(in_w, w_w) <= 16:
        return 'acc64'
    raise ValueError(
        f'No accumulator tag registered for AIE generation "{device.generation}" with '
        f'input {in_w}-bit and weight {w_w}-bit precisions.'
    )


def _extract_rounding(src):
    if hasattr(src, 'rounding_mode'):
        return src.rounding_mode or RoundingMode.TRN
    if hasattr(src, 'rounding'):
        return src.rounding
    return RoundingMode.TRN


def _aie_rounding_token(source) -> str:
    mode = _extract_rounding(source)
    token = ROUNDING_TOKEN_MAP.get(mode)
    if token is None:
        raise ValueError(f'Unsupported rounding mode {mode} for AIE kernel.')
    return token


def resolve_scalars(ctx: LayerResolveContext) -> None:
    scalars = ctx.attributes.scalars
    if ctx.policy.requires_numeric and 'shift' not in scalars:
        raise RuntimeError(f'{ctx.layer_name}: missing resolved numeric scalar "shift"')

    parallelism = ctx.parallelism()
    scalars['padded_independent_extent'] = int(parallelism.padded_independent_extent)
    scalars['real_independent_extent'] = int(parallelism.independent_extent)
    scalars['padded_in_features'] = int(parallelism.padded_in_features)
    scalars['padded_out_features'] = int(parallelism.padded_out_features)
    scalars['batch_size'] = int(ctx.node.inputs[0].shape[0])
    scalars['io_shapes'] = _resolve_io_shapes(ctx)


def merge_config_layers(config: dict, layer_name: str, layer_class: str) -> dict:
    h = config.get('HLSConfig', {}) or {}
    merged = {}
    for scope in (
        h.get('AIE', {}),
        h.get('LayerType', {}).get(layer_class, {}),
        h.get('LayerName', {}).get(layer_name, {}),
    ):
        if isinstance(scope, dict):
            for k, v in scope.items():
                if isinstance(v, dict):
                    merged.setdefault(k, {}).update(v)
                else:
                    merged[k] = v
    return merged


def register_layer_policy(name: str, policy: LayerPolicy) -> None:
    LAYER_RESOLVE_REGISTRY[name] = policy


def get_layer_policy(layer_class: str) -> Optional[LayerPolicy]:
    return LAYER_RESOLVE_REGISTRY.get(layer_class)


LAYER_RESOLVE_REGISTRY: Dict[str, LayerPolicy] = {
    'Input': LayerPolicy(
        namespaces=(),
        resolvers=(),
        requires_numeric=False,
    ),
    'Dense': LayerPolicy(
        namespaces=('numeric', 'tiling', 'parallelism', 'slices', 'flags', 'pack', 'scalars'),
        requires_numeric=True,
        resolvers=(
            resolve_numeric,
            resolve_tiling,
            resolve_parallelism,
            resolve_flags,
            resolve_io_route,
            resolve_io_view,
            resolve_staging,
            resolve_pack,
            resolve_placement,
            resolve_scalars,
        ),
    ),
    'Activation': LayerPolicy(
        namespaces=('numeric', 'slices', 'flags', 'scalars'),
        requires_numeric=False,
        resolvers=(
            resolve_numeric,
            resolve_parallelism,
            resolve_flags,
            resolve_io_view,
            resolve_placement,
            resolve_scalars,
        ),
    ),
}


__all__ = [
    'LayerPolicy',
    'LayerResolveContext',
    'NumericBundle',
    'ParallelismResult',
    'get_layer_policy',
    'merge_config_layers',
    'register_layer_policy',
]
