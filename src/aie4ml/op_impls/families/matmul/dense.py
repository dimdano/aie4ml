from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ....aie_types import FloatIntent
from ....ir.graph import OpImplInstance, OpNode, input_tensor_for_role
from ....passes.utils import sanitize_identifier
from ...base import OpImplFootprint, OpImplVariant
from ...common_types import PortBinding, PortMap
from ...registry import register_variant
from ...utils import ParallelismConfig, parse_directives
from ...utils.precision import (
    aie_rounding_token,
    infer_accumulator_tag,
    resolve_accumulator_output_shift,
    resolve_exact_storage_dtype,
)
from .common import (
    MICROTILE_OPTIONS,
    describe_family_lhs_staging,
    describe_family_output_staging,
    np_bias_dtype_for_spec,
    np_dtype_for_spec,
    pack_as_float,
    pack_mmul_rhs_matrix,
    pack_vector_by_n_slice,
    quantize_to_int,
    select_generation_key,
)
from .config import DenseConfig, DenseFlags
from .resolver import (
    _build_io_views,
    _resolve_numeric,
    _resolve_output_scale_shift,
    _resolve_parallelism,
    _resolve_tile_cfg,
)

_SUPPORTED_INT_WIDTH_COMBOS = frozenset({(8, 8), (16, 8), (16, 16)})


class _BaseDenseMatmulVariant(OpImplVariant):
    """Unregistered shared base for Dense and Matmul variants."""

    def build_template_params(self, node, config):
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        lhs_view = config.io_views[lhs_tensor.name]
        output_view = config.io_views[node.outputs[0].name]
        params = {f: getattr(config, f) for f in config.__dataclass_fields__}
        params.update(
            full_outer=lhs_view.compacted_full_outer,
            full_inner_lhs=lhs_view.full_inner,
            full_inner_rhs=output_view.full_inner,
            tile_inner_lhs=lhs_view.tile_inner,
            tile_inner_rhs=output_view.tile_inner,
            tile_inner_lhs_raw=lhs_view.tile_raw_inner,
            tile_inner_rhs_raw=output_view.tile_raw_inner,
        )
        return params

    def describe_input_staging(self, _node, config, tensor_name, port, buf_dims=None, _producer=None):
        view = config.io_views[tensor_name]
        return describe_family_lhs_staging(view, config.microtiling, port, buf_dims)

    def describe_output_staging(self, _node, config, tensor_name, port, buf_dims=None):
        view = config.io_views[tensor_name]
        return describe_family_output_staging(view, config.microtiling, port, buf_dims)

    def microtiling_options(self, generation: str, query) -> List[Tuple[int, int, int]]:
        return list(MICROTILE_OPTIONS.get(select_generation_key(generation), {}).get(tuple(query), []))

    def output_staging_contract(self, _node, _config, _tensor_name: str):
        return 'inner'


@register_variant
class DenseOpImplVariant(_BaseDenseMatmulVariant):
    variant_id = 'dense.b.r.v1'
    op_type = 'dense'
    graph_header = 'dense_bias_relu_graph.h'
    graph_name = 'dense_bias_relu_graph'
    param_template = 'dense_bias_relu'
    plevel = 10

    def matches(self, node: OpNode, device) -> bool:
        if device.generation not in ('AIE-ML', 'AIE-MLV2'):
            return False
        lhs = input_tensor_for_role(node, 'lhs')
        rhs = input_tensor_for_role(node, 'rhs')
        if isinstance(lhs.precision, FloatIntent):
            return True
        lhs_p = resolve_exact_storage_dtype(lhs.precision, namespace='lhs', layer_name=node.name)
        rhs_p = resolve_exact_storage_dtype(rhs.precision, namespace='rhs', layer_name=node.name)
        return (lhs_p.width, rhs_p.width) in _SUPPORTED_INT_WIDTH_COMBOS

    def resolve(self, node: OpNode, device, directives=None) -> DenseConfig:
        io_route, _, _ = parse_directives(directives)
        precision = _resolve_numeric(node, device)
        microtiling = _resolve_tile_cfg(node, device, precision['lhs'], precision['rhs'])
        tiling = _resolve_parallelism(node, device, microtiling, precision)
        io_views = _build_io_views(node, microtiling, tiling)

        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        lhs_perm = io_views[lhs_tensor.name].perm
        is_float = isinstance(lhs_tensor.precision, FloatIntent)

        shift = (
            0
            if is_float
            else resolve_accumulator_output_shift(lhs_tensor.precision, node.outputs[0].precision, rhs_tensor.precision)
        )
        shift += _resolve_output_scale_shift(node, is_float=is_float)

        fused_act = node.traits.get('fused_activation')
        use_relu = ((fused_act.data.get('activation') if fused_act else '') or '').lower() == 'relu'

        return DenseConfig(
            precision=precision,
            parallelism=ParallelismConfig(cas_length=tiling.cas_length, cas_num=tiling.cas_num),
            microtiling=microtiling,
            io_views=io_views,
            io_route=io_route,
            shift=shift,
            accumulator_tag=infer_accumulator_tag(device, None, None, precision['acc']),
            rounding_mode='conv_even' if is_float else aie_rounding_token(precision['output']),
            flags=DenseFlags(
                use_relu=use_relu,
                transpose_lhs=bool(lhs_perm is not None and lhs_perm[-1] != (len(lhs_perm) - 1)),
                use_bias=bool(node.metadata.get('use_bias')),
            ),
        )

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        p = inst.config
        input_tensor = inst.node.inputs[0]
        weight_tensor = inst.node.inputs[1]
        bias_tensor = inst.node.inputs[2] if len(inst.node.inputs) > 2 else None
        lhs_view = p.io_views[input_tensor.name]
        output_view = p.io_views[inst.node.outputs[0].name]

        wi = weight_tensor.precision
        if isinstance(wi, FloatIntent):
            W = pack_as_float(weight_tensor.data, wi.format)
            b = np.asarray(bias_tensor.data, dtype=np.float32) if bias_tensor is not None else None
        else:
            W = quantize_to_int(
                weight_tensor.data,
                wi.frac,
                wi.width,
                signed=wi.signed,
                rounding_mode=wi.rounding,
                saturation_mode=wi.saturation,
            )
            if bias_tensor is not None:
                bi = bias_tensor.precision
                accum_frac = input_tensor.precision.frac + wi.frac
                b = quantize_to_int(
                    bias_tensor.data,
                    accum_frac,
                    32,
                    signed=bi.signed,
                    rounding_mode=bi.rounding,
                    saturation_mode=bi.saturation,
                )
            else:
                b = None

        W = np.asarray(W)
        if W.ndim < 2:
            raise ValueError(f'{inst.name}: weight matrix must have at least 2 dimensions, got {W.ndim}.')
        n_in = int(W.shape[-2])
        n_out = int(W.shape[-1])

        packed_W = pack_mmul_rhs_matrix(
            W,
            K=n_in,
            N=n_out,
            K_slice=lhs_view.tile_inner,
            N_slice=output_view.tile_inner,
            microtile_k=p.microtiling.microtile_k,
            microtile_n=p.microtiling.microtile_n,
            cas_length=p.parallelism.cas_length,
            cas_num=p.parallelism.cas_num,
            dtype=np_dtype_for_spec(p.precision['rhs']),
        )
        packed_B = (
            pack_vector_by_n_slice(
                b,
                N=n_out,
                N_slice=output_view.tile_inner,
                cas_num=p.parallelism.cas_num,
                dtype=np_bias_dtype_for_spec(p.precision['bias']),
            )
            if b is not None
            else None
        )
        return {'packed_weights': packed_W, 'packed_bias': packed_B}

    def footprint(self, _node, config) -> OpImplFootprint:
        return OpImplFootprint(
            width=config.parallelism.cas_length,
            height=config.parallelism.cas_num,
            extras={'keepout_left': 1},
        )

    def get_artifacts(self, inst: OpImplInstance):
        inst_name = sanitize_identifier(inst.name)
        p = inst.config
        output_view = p.io_views[inst.node.outputs[0].name]
        artifacts = [
            {
                'name': 'weights',
                'kind': '2d',
                'storage': 'rom',
                'array': inst.artifacts['packed_weights'],
                'dtype': p.precision['rhs'].c_type,
                'storage_dtype': p.precision['rhs'].storage_dtype,
                'filename': f'weights_{inst_name}.h',
                'port': 'wts',
            }
        ]
        packed_bias = inst.artifacts.get('packed_bias')
        if packed_bias is None:
            # Dense graph always exposes a bias RTP port; feed explicit zeros for biasless layers.
            packed_bias = np.zeros(
                (int(p.parallelism.cas_num), output_view.tile_inner),
                dtype=np_bias_dtype_for_spec(p.precision['bias']),
            )
        artifacts.append(
            {
                'name': 'bias',
                'kind': '1d',
                'storage': 'rom',
                'array': packed_bias,
                'dtype': p.precision['bias'].c_type,
                'storage_dtype': p.precision['bias'].storage_dtype,
                'filename': f'bias_{inst_name}.h',
                'port': 'bias',
            }
        )
        return artifacts

    def validate_config(self, node: OpNode, config: DenseConfig, _device) -> None:
        if config.shift < 0:
            raise ValueError(f'{node.name}: dense accumulator output shift must be non-negative, got {config.shift}.')

    def build_ports(self, node: OpNode, config: DenseConfig):
        n_in = int(config.parallelism.cas_length)
        n_out = int(config.parallelism.cas_num)
        data_inputs = [t for t in node.inputs if not t.is_parameter]
        return PortMap(
            inputs={t.name: PortBinding(group=f'in{i+1}', count=n_in) for i, t in enumerate(data_inputs)},
            outputs={t.name: PortBinding(group=f'out{i+1}', count=n_out) for i, t in enumerate(node.outputs)},
        )
