from __future__ import annotations

import math
from typing import Any, Dict

from ....aie_types import AIEDataType, FloatIntent
from ....ir.graph import OpImplInstance, OpNode, input_tensor_for_role
from ....passes.utils import sanitize_identifier
from ...base import OpImplFootprint, OpImplVariant
from ...common_types import PortBinding, PortMap
from ...registry import register_variant
from ...utils import (
    ParallelismConfig,
    build_partition_views,
    decompose_shape,
    find_tile_split,
    parse_directives,
    require_power_of_two,
)
from ...utils.io import view_shape
from ...utils.precision import (
    aie_rounding_token,
    resolve_exact_storage_dtype,
    storage_bytes_for_spec,
    to_quant_intent,
)
from ..elementwise.common import describe_elementwise_staging
from .common import (
    BETA_FRAC_BITS,
    DEFAULT_ISQRT_NR_ITERS,
    DEFAULT_USE_AIE_INVSQRT,
    GAMMA_FRAC_BITS,
    layernorm_vec_size,
    pack_layernorm_param,
)
from .config import LayerNormConfig


def _resolve_eps_q0(node_name: str, metadata: dict, input_frac: int) -> int:
    if 'epsilon' in metadata:
        epsilon = float(metadata['epsilon'])
        if epsilon < 0.0:
            raise ValueError(f'{node_name}: LayerNormalization epsilon must be non-negative, got {epsilon}.')
        eps_q0_f = epsilon * float(1 << (2 * int(input_frac)))
        eps_q0 = int(round(eps_q0_f))
        if not math.isclose(eps_q0_f, float(eps_q0), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f'{node_name}: epsilon={epsilon} is not exactly representable as integer EPS_Q0 '
                f'for input frac={input_frac} (epsilon / input_scale^2 = {eps_q0_f}).'
            )
        if eps_q0 < 1:
            raise ValueError(
                f'{node_name}: epsilon={epsilon} maps to EPS_Q0={eps_q0}; '
                'integer LayerNorm requires a positive variance floor.'
            )
        return eps_q0
    eps_q0 = int(metadata.get('eps_q0', 1))
    if eps_q0 < 1:
        raise ValueError(f'{node_name}: eps_q0 must be positive, got {eps_q0}.')
    return eps_q0


@register_variant
class LayerNormI8OpImplVariant(OpImplVariant):
    variant_id = 'layer_norm.i8.v1'
    op_type = 'layer_norm'
    graph_header = 'layer_norm_graph.h'
    graph_name = 'layer_norm_graph'
    param_template = 'layer_norm'
    plevel = 10

    def matches(self, node: OpNode, device) -> bool:
        if device.generation not in ('AIE-ML', 'AIE-MLV2'):
            return False
        in_tensor = input_tensor_for_role(node, 'lhs')
        if isinstance(in_tensor.precision, FloatIntent):
            return False
        in_prec = resolve_exact_storage_dtype(in_tensor.precision, namespace='lhs', layer_name=node.name)
        out_prec = resolve_exact_storage_dtype(node.outputs[0].precision, namespace='output', layer_name=node.name)
        return int(in_prec.width) == 8 and bool(in_prec.signed) and int(out_prec.width) == 8 and bool(out_prec.signed)

    def resolve(self, node: OpNode, device, directives=None) -> LayerNormConfig:
        io_route, input_contracts, parallel_cfg = parse_directives(directives)

        in_tensor = input_tensor_for_role(node, 'lhs')
        out_tensor = node.outputs[0]

        precision = {
            'lhs': resolve_exact_storage_dtype(in_tensor.precision, namespace='lhs', layer_name=node.name),
            'output': resolve_exact_storage_dtype(out_tensor.precision, namespace='output', layer_name=node.name),
            'gamma': AIEDataType(format='int16', frac=GAMMA_FRAC_BITS),
            'beta': AIEDataType(format='int16', frac=BETA_FRAC_BITS),
        }

        in_shape = tuple(int(x) for x in view_shape(node, in_tensor, 'inputs'))
        full_inner, outer_prefix, last_outer = decompose_shape(in_shape)

        vec_size = layernorm_vec_size(precision['lhs'], device)
        if full_inner % vec_size != 0:
            raise ValueError(
                f'{node.name}: full_inner={full_inner} must be a multiple of vec_size={vec_size}; '
                'pad the inner dimension before LayerNorm.'
            )

        in_bpp = storage_bytes_for_spec(precision['lhs'])
        out_bpp = storage_bytes_for_spec(precision['output'])
        gamma_bpp = storage_bytes_for_spec(precision['gamma'])
        beta_bpp = storage_bytes_for_spec(precision['beta'])

        cas_num, tile_outer = find_tile_split(
            partition_size=last_outer,
            max_rows=max(1, int(device.rows)),
            bank_bytes=int(device.bank_mem_bytes),
            tile_bytes_fn=lambda to: max(
                outer_prefix * to * full_inner * in_bpp,
                outer_prefix * to * full_inner * out_bpp,
                full_inner * gamma_bpp,
                full_inner * beta_bpp,
            ),
            parallel_cfg=parallel_cfg,
            input_contracts=input_contracts,
            primary_tensor_name=in_tensor.name,
            contract='outer',
            descending=True,
        )

        io_views = build_partition_views(
            node,
            [in_tensor],
            [out_tensor],
            full_inner=full_inner,
            full_outer=last_outer,
            tile_inner=full_inner,
            tile_outer=tile_outer,
            tile_inner_raw=full_inner,
            tile_outer_raw=tile_outer,
        )

        out_shift = int(to_quant_intent(out_tensor.precision).frac)

        return LayerNormConfig(
            precision=precision,
            parallelism=ParallelismConfig(cas_num=int(cas_num)),
            rows=int(outer_prefix * tile_outer),
            cols=int(full_inner),
            vec_size=int(vec_size),
            gamma_shift=int(GAMMA_FRAC_BITS),
            out_shift=out_shift,
            eps_q0=_resolve_eps_q0(node.name, node.metadata, int(precision['lhs'].frac)),
            isqrt_nr_iters=int(DEFAULT_ISQRT_NR_ITERS),
            use_aie_invsqrt=bool(DEFAULT_USE_AIE_INVSQRT),
            rounding_mode=aie_rounding_token(precision['output']),
            io_views=io_views,
            io_route=io_route,
        )

    def validate_config(self, node: OpNode, config: LayerNormConfig, _device) -> None:
        require_power_of_two(f'{node.name}: cols', config.cols)
        if not (0 <= config.out_shift <= 15):
            raise ValueError(
                f'{node.name}: out_shift={config.out_shift} must be in [0, 15]; '
                'integer LayerNorm cannot left-shift or exceed NORM_SHIFT=15.'
            )

    def build_template_params(self, _node: OpNode, config: LayerNormConfig):
        return {f: getattr(config, f) for f in config.__dataclass_fields__}

    def describe_input_staging(self, _node, config, tensor_name, port, buf_dims=None, _producer=None):
        return describe_elementwise_staging(config.io_views[tensor_name], port, 'read', 'outer', buf_dims)

    def describe_output_staging(self, _node, config, tensor_name, port, buf_dims=None):
        return describe_elementwise_staging(config.io_views[tensor_name], port, 'write', 'outer', buf_dims)

    def output_staging_contract(self, _node, _config: LayerNormConfig, _tensor_name: str):
        return 'outer'

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        p = inst.config
        gamma_tensor = input_tensor_for_role(inst.node, 'gamma')
        beta_tensor = input_tensor_for_role(inst.node, 'beta')
        cas_num = int(p.parallelism.cas_num)
        return {
            'packed_gamma': pack_layernorm_param(
                gamma_tensor.data,
                name='gamma',
                full_inner=int(p.cols),
                frac=int(p.precision['gamma'].frac),
                cas_num=cas_num,
            ),
            'packed_beta': pack_layernorm_param(
                beta_tensor.data,
                name='beta',
                full_inner=int(p.cols),
                frac=int(p.precision['beta'].frac),
                cas_num=cas_num,
            ),
        }

    def get_artifacts(self, inst: OpImplInstance):
        inst_name = sanitize_identifier(inst.name)
        p = inst.config
        return [
            {
                'name': 'gamma',
                'kind': '1d',
                'storage': 'rom',
                'array': inst.artifacts['packed_gamma'],
                'dtype': p.precision['gamma'].c_type,
                'storage_dtype': p.precision['gamma'].storage_dtype,
                'filename': f'gamma_{inst_name}.h',
                'port': 'gamma',
            },
            {
                'name': 'beta',
                'kind': '1d',
                'storage': 'rom',
                'array': inst.artifacts['packed_beta'],
                'dtype': p.precision['beta'].c_type,
                'storage_dtype': p.precision['beta'].storage_dtype,
                'filename': f'beta_{inst_name}.h',
                'port': 'beta',
            },
        ]

    def footprint(self, node: OpNode, config: LayerNormConfig) -> OpImplFootprint:
        return OpImplFootprint(width=1, height=int(config.parallelism.cas_num), extras={'keepout_left': 1})

    def build_ports(self, node: OpNode, config: LayerNormConfig):
        in_tensor = input_tensor_for_role(node, 'lhs')
        n = int(config.parallelism.cas_num)
        return PortMap(
            inputs={in_tensor.name: PortBinding(group='in1', count=n)},
            outputs={node.outputs[0].name: PortBinding(group='out1', count=n)},
        )
