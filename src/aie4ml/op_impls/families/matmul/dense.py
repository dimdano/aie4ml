from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ....ir.graph import OpImplInstance, OpNode, input_tensor_for_role
from ....passes.utils import sanitize_identifier
from ...base import OpImplFootprint, OpImplVariant
from ...registry import register_variant
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
    validate_family_tile_contract,
)
from .config import DenseConfig


class _BaseDenseMatmulVariant(OpImplVariant):
    """Unregistered shared base for Dense and Matmul variants.

    Holds the methods that are identical across both op types.
    Each subclass is responsible for its own class-var declarations
    (variant_id, op_type, supported_precisions, etc.) and any
    method overrides specific to that op.
    """

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

    def describe_input_staging(self, node, config, tensor_name, port, buf_dims=None, producer=None):
        view = config.io_views[tensor_name]
        return describe_family_lhs_staging(view, config.microtiling, port, buf_dims)

    def describe_output_staging(self, node, config, tensor_name, port, buf_dims=None):
        view = config.io_views[tensor_name]
        return describe_family_output_staging(view, config.microtiling, port, buf_dims)

    def microtiling_options(self, generation: str, query) -> List[Tuple[int, int, int]]:
        return list(MICROTILE_OPTIONS.get(select_generation_key(generation), {}).get(tuple(query), []))

    def output_staging_contract(self, node, config, tensor_name: str):
        return 'inner'


@register_variant
class DenseOpImplVariant(_BaseDenseMatmulVariant):
    variant_id = 'dense.b.r.v1'
    op_type = 'dense'
    graph_header = 'dense_bias_relu_graph.h'
    graph_name = 'dense_bias_relu_graph'
    param_template = 'dense_bias_relu'
    supported_generations = ('AIE-ML', 'AIE-MLV2')
    supported_precisions = (
        {'lhs': 8, 'rhs': 8, 'output': 8, 'acc': 32, 'bias': 32},
        {'lhs': 8, 'rhs': 8, 'output': 16, 'acc': 32, 'bias': 32},
        {'lhs': 8, 'rhs': 8, 'output': 32, 'acc': 32, 'bias': 32},
        {'lhs': 16, 'rhs': 8, 'output': 8, 'acc': 32, 'bias': 32},
        {'lhs': 16, 'rhs': 16, 'output': 16, 'acc': 64, 'bias': 32},
        {'lhs': 16, 'rhs': 16, 'output': 32, 'acc': 64, 'bias': 32},
        {'lhs': 16, 'rhs': 16, 'output': 16, 'acc': 32, 'bias': 32, 'lhs_c_type': 'bfloat16'},
        {'lhs': 32, 'rhs': 32, 'output': 32, 'acc': 32, 'bias': 32, 'lhs_c_type': 'float'},
    )
    supported_input_modes = ('direct', 'memtile', 'plio', 'auto')
    supported_output_modes = ('direct', 'memtile', 'plio', 'auto')

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        from ....aie_types import FloatFormat, FloatIntent

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
        if isinstance(wi, FloatIntent) and wi.format == FloatFormat.BF16:
            packed_W = (packed_W.astype(np.uint32) << 16).view(np.float32)
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

    def validate_config(self, node: OpNode, config: DenseConfig, device) -> None:
        p = config
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        lhs_view = p.io_views[lhs_tensor.name]
        output_view = p.io_views[node.outputs[0].name]
        validate_family_tile_contract(
            node_name=node.name,
            precision=p.precision,
            parallelism=p.parallelism,
            microtiling=p.microtiling,
            lhs_view=lhs_view,
            output_view=output_view,
            bank_bytes=int(device.bank_mem_bytes),
        )

    def footprint(self, node, config) -> OpImplFootprint:
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
                'filename': f'weights_{inst_name}.h',
                'port': 'wts',
            }
        ]
        packed_bias = inst.artifacts.get('packed_bias')
        if packed_bias is None:
            # The current dense graph always exposes a bias RTP port. For biasless
            # layers we feed an explicit zero bias matching the fixed kernel
            # interface instead of trying to specialize the graph signature.
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
                'filename': f'bias_{inst_name}.h',
                'port': 'bias',
            }
        )
        return artifacts

    def build_ports(self, node: OpNode, config: DenseConfig):
        return super().build_ports(node, int(config.parallelism.cas_length), int(config.parallelism.cas_num))
