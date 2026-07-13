from __future__ import annotations

from typing import Any, Dict

from ....aie_types import AIEDataType, FloatIntent
from ....ir.graph import OpImplInstance, OpNode, input_tensor_for_role
from ...base import OpImplFootprint, OpImplVariant
from ...common_types import PortBinding, PortMap
from ...registry import register_variant
from ...utils import (
    ParallelismConfig,
    build_io_views,
    describe_partition_staging,
    extract_inner_outer,
    find_tile_split,
    parse_directives,
)
from ...utils.io import view_shape
from ...utils.precision import resolve_exact_storage_dtype, storage_bytes_for_spec
from .common import DEFAULT_INV_SHIFT, infer_hccs_param_sets, pack_hccs_params, softmax_vec_size, validate_hccs_params
from .config import SoftmaxConfig


def _parse_hccs_directives(node_name: str, directives) -> dict:
    directives = directives or {}
    approximation = str(directives.get('approximation', 'hccs')).lower()
    if approximation != 'hccs':
        raise ValueError(f'{node_name}: only approximation="hccs" is supported for integer Softmax.')
    hccs = dict(directives.get('hccs', {}) or {})
    missing = [name for name in ('B', 'S', 'Dmax') if name not in hccs]
    if missing:
        raise ValueError(f'{node_name}: HCCS Softmax directives missing {", ".join(missing)}.')
    return hccs


@register_variant
class SoftmaxHccsI8OpImplVariant(OpImplVariant):
    """HCCS integer Softmax surrogate.

    Implements Head-Calibrated Clipped-Linear Softmax, an integer attention-softmax
    surrogate using calibrated B/S/Dmax parameters instead of exponentials.
    This variant is intended for QAT/calibrated models and is not a drop-in
    replacement for generic floating-point ONNX Softmax. See https://arxiv.org/pdf/2604.02292v1
    """

    variant_id = 'softmax.hccs.i8.v1'
    op_type = 'softmax'
    graph_header = 'softmax_graph.h'
    graph_name = 'softmax_hccs_graph'
    param_template = 'softmax'
    plevel = 10

    def matches(self, node: OpNode, device) -> bool:
        if device.generation not in ('AIE-ML', 'AIE-MLV2'):
            return False
        in_tensor = input_tensor_for_role(node, 'lhs')
        if isinstance(in_tensor.precision, FloatIntent):
            return False
        directives = node.directives or {}
        if str(directives.get('approximation', 'hccs')).lower() != 'hccs':
            return False
        in_prec = resolve_exact_storage_dtype(in_tensor.precision, namespace='lhs', layer_name=node.name)
        return in_prec.format == 'int8'

    def resolve(self, node: OpNode, device, directives=None) -> SoftmaxConfig:
        io_route, input_contracts, parallel_cfg = parse_directives(directives)
        hccs = _parse_hccs_directives(node.name, directives)

        input_scale = float(node.trait_data('input_scale').get('scale', 1.0))
        if abs(input_scale - 1.0) > 1e-9:
            raise ValueError(
                f'{node.name}: softmax.hccs.i8.v1 cannot apply a runtime input_scale={input_scale}; '
                'bake the softmax temperature into the HCCS B/S/Dmax calibration, or lower to a '
                'float softmax variant that applies input_scale at runtime.'
            )

        in_tensor = input_tensor_for_role(node, 'lhs')
        out_tensor = node.outputs[0]

        precision = {
            'lhs': resolve_exact_storage_dtype(in_tensor.precision, namespace='lhs', layer_name=node.name),
            'output': resolve_exact_storage_dtype(out_tensor.precision, namespace='output', layer_name=node.name),
            'B': AIEDataType(format='int16'),
            'S': AIEDataType(format='int8'),
            'Dmax': AIEDataType(format='uint8'),
        }

        in_shape = tuple(int(x) for x in view_shape(node, in_tensor, 'inputs'))
        full_inner, outer_prefix, last_outer = extract_inner_outer(in_shape)
        vec_size = softmax_vec_size(precision['lhs'], device)
        if full_inner % vec_size != 0:
            raise ValueError(
                f'{node.name}: softmax axis length {full_inner} must be a multiple of vec_size={vec_size}; '
                'pad the softmax dimension before lowering.'
            )

        cas_length = int(parallel_cfg.get('cas_length', 1))
        if cas_length != 1:
            raise ValueError(f'{node.name}: HCCS Softmax requires cas_length=1, got {cas_length}.')

        in_bpp = storage_bytes_for_spec(precision['lhs'])
        out_bpp = storage_bytes_for_spec(precision['output'])
        cas_num, tile_outer = find_tile_split(
            partition_size=last_outer,
            max_rows=max(1, int(device.rows)),
            bank_bytes=int(device.bank_mem_bytes),
            tile_bytes_fn=lambda to: max(
                outer_prefix * to * full_inner * in_bpp,
                outer_prefix * to * full_inner * out_bpp,
                full_inner * 2,
            ),
            parallel_cfg=parallel_cfg,
            input_contracts=input_contracts,
            primary_tensor_name=in_tensor.name,
            contract='outer',
        )

        io_views = build_io_views(
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

        return SoftmaxConfig(
            precision=precision,
            parallelism=ParallelismConfig(cas_num=int(cas_num)),
            param_sets=int(infer_hccs_param_sets(hccs)),
            vec_size=int(vec_size),
            inv_shift=int(hccs.get('inv_shift', DEFAULT_INV_SHIFT)),
            use_clb=bool(hccs.get('use_clb', False)),
            io_views=io_views,
            io_route=io_route,
            hccs=hccs,
        )

    def validate_config(self, node: OpNode, config: SoftmaxConfig, _device) -> None:
        out_format = config.precision['output'].format
        if out_format not in ('uint8', 'int16'):
            raise ValueError(f'{node.name}: softmax.hccs.i8.v1 requires uint8 or int16 output, got {out_format!r}.')
        if config.param_sets != 1:
            raise ValueError(
                f'{node.name}: softmax.hccs.i8.v1 does not support multi-head param_sets={config.param_sets}; '
                'lower attention to one Softmax op per head.'
            )
        if not (1 <= int(config.inv_shift) <= 30):
            raise ValueError(f'{node.name}: HCCS Softmax inv_shift must be in [1, 30], got {config.inv_shift}.')
        in_view = config.io_views[input_tensor_for_role(node, 'lhs').name]
        validate_hccs_params(config.hccs, cols=int(in_view.full_inner), param_sets=int(config.param_sets))

    def build_template_params(self, node: OpNode, config: SoftmaxConfig):
        in_tensor = input_tensor_for_role(node, 'lhs')
        in_view = config.io_views[in_tensor.name]
        params = {f: getattr(config, f) for f in config.__dataclass_fields__}
        params.update(
            rows=int(in_view.compacted_tile_outer),
            cols=int(in_view.full_inner),
        )
        params['packed_hccs'] = pack_hccs_params(
            config.hccs,
            param_sets=int(config.param_sets),
            cols=int(in_view.full_inner),
            cas_num=int(config.parallelism.cas_num),
        )
        return params

    def describe_input_staging(self, _node, config, tensor_name, port, buf_dims=None, _producer=None):
        return describe_partition_staging(config.io_views[tensor_name], port, 'read', 'outer', buf_dims)

    def describe_output_staging(self, _node, config, tensor_name, port, buf_dims=None):
        return describe_partition_staging(config.io_views[tensor_name], port, 'write', 'outer', buf_dims)

    def output_staging_contract(self, _node, _config: SoftmaxConfig, _tensor_name: str):
        return 'outer'

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        return {}

    def get_artifacts(self, inst: OpImplInstance):
        return []

    def footprint(self, node: OpNode, config: SoftmaxConfig) -> OpImplFootprint:
        return OpImplFootprint(width=1, height=int(config.parallelism.cas_num), extras={'keepout_left': 1})

    def build_ports(self, node: OpNode, config: SoftmaxConfig):
        in_tensor = input_tensor_for_role(node, 'lhs')
        n = int(config.parallelism.cas_num)
        return PortMap(
            inputs={in_tensor.name: PortBinding(group='in1', count=n)},
            outputs={node.outputs[0].name: PortBinding(group='out1', count=n)},
        )
