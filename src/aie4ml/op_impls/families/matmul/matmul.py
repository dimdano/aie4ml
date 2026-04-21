from __future__ import annotations

from ....aie_types import FloatIntent
from ....ir.graph import OpImplInstance, input_role, input_tensor_for_role
from ...base import OpImplFootprint
from ...common_types import PortBinding, PortMap
from ...registry import register_variant
from .common import (
    describe_family_lhs_staging,
    describe_family_rhs_staging,
    validate_family_tile_contract,
)
from .config import MatmulConfig
from .dense import _BaseDenseMatmulVariant

_MATMUL_STACK_OVERHEAD_BYTES = 1024  # kernel stack reserved for matmul RHS packing


@register_variant
class MatmulOpImplVariant(_BaseDenseMatmulVariant):
    variant_id = 'matmul.v1'
    op_type = 'matmul'
    graph_header = 'matmul_graph.h'
    graph_name = 'matmul_graph'
    param_template = 'matmul'
    supported_generations = ('AIE-ML', 'AIE-MLV2')
    supported_precisions = (
        {'lhs': 8, 'rhs': 8, 'output': 8, 'acc': 32, 'rhs_c_type': 'int8_t'},
        {'lhs': 8, 'rhs': 8, 'output': 16, 'acc': 32, 'rhs_c_type': 'int8_t'},
        {'lhs': 8, 'rhs': 8, 'output': 32, 'acc': 32, 'rhs_c_type': 'int8_t'},
        {'lhs': 16, 'rhs': 8, 'output': 8, 'acc': 32, 'rhs_c_type': 'int8_t'},
        {'lhs': 16, 'rhs': 16, 'output': 16, 'acc': 64, 'rhs_c_type': 'int16_t'},
        {'lhs': 16, 'rhs': 16, 'output': 32, 'acc': 64, 'rhs_c_type': 'int16_t'},
        {'lhs': 16, 'rhs': 16, 'output': 16, 'acc': 32, 'lhs_c_type': 'bfloat16', 'rhs_c_type': 'bfloat16'},
        {'lhs': 32, 'rhs': 32, 'output': 32, 'acc': 32, 'lhs_c_type': 'float', 'rhs_c_type': 'float'},
    )
    supported_input_modes = ('direct', 'memtile', 'plio', 'auto')
    supported_output_modes = ('direct', 'memtile', 'plio', 'auto')

    def describe_input_staging(self, node, config, tensor_name, port, buf_dims=None, producer=None):
        view = config.io_views[tensor_name]
        role = input_role(node, tensor_name)
        if role == 'rhs':
            return describe_family_rhs_staging(view, config.microtiling, config.parallelism, port, buf_dims)
        return describe_family_lhs_staging(view, config.microtiling, port, buf_dims)

    def pack(self, inst: OpImplInstance):
        return {}

    def get_artifacts(self, inst: OpImplInstance):
        return []

    def validate_config(self, node, config: MatmulConfig, device) -> None:
        p = config
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        lhs_view = p.io_views[lhs_tensor.name]
        output_view = p.io_views[node.outputs[0].name]

        rhs_prec = p.precision['rhs']
        rhs_is_float = isinstance(rhs_tensor.precision, FloatIntent)
        if not rhs_is_float and not bool(rhs_prec.signed):
            raise ValueError(f'{node.name}: matmul RHS must use a signed integer precision.')

        rhs_shape = tuple(int(x) for x in p.io_views[rhs_tensor.name].logical)
        if len(rhs_shape) < 2:
            raise ValueError(f'{node.name}: matmul RHS must be rank >=2, got {len(rhs_shape)}.')
        validate_family_tile_contract(
            node_name=node.name,
            precision=p.precision,
            parallelism=p.parallelism,
            microtiling=p.microtiling,
            lhs_view=lhs_view,
            output_view=output_view,
            bank_bytes=int(device.bank_mem_bytes),
            rhs_overhead_bytes=_MATMUL_STACK_OVERHEAD_BYTES,
        )

    def footprint(self, node, config) -> OpImplFootprint:
        return OpImplFootprint(
            width=config.parallelism.cas_length,
            height=config.parallelism.cas_num,
            extras={'keepout_left': 1},
        )

    def build_ports(self, node, config: MatmulConfig):
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        return PortMap(
            inputs={
                lhs_tensor.name: PortBinding(group='inA', count=int(config.parallelism.cas_length)),
                rhs_tensor.name: PortBinding(
                    group='inB',
                    count=int(config.parallelism.cas_length) * int(config.parallelism.cas_num),
                ),
            },
            outputs={node.outputs[0].name: PortBinding(group='outC', count=int(config.parallelism.cas_num))},
        )
