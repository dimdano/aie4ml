from __future__ import annotations

import math
from typing import Any, Dict

from ....ir.graph import OpImplInstance, OpNode, input_tensor_for_role
from ...base import OpImplFootprint, OpImplVariant
from ...registry import register_variant
from .common import describe_elementwise_staging, validate_elementwise_tile_contract
from .config import AddConfig


@register_variant
class AddOpImplVariant(OpImplVariant):
    variant_id = 'add.v1'
    op_type = 'add'
    graph_header = 'elementwise_add_graph.h'
    graph_name = 'elementwise_add_graph'
    param_template = 'elementwise_add'
    supported_generations = ('AIE-ML', 'AIE-MLV2')
    supported_precisions = (
        {'lhs': 8, 'rhs': 8, 'output': 8},
        {'lhs': 16, 'rhs': 16, 'output': 16},
        {'lhs': 32, 'rhs': 32, 'output': 32},
        {'lhs': 16, 'rhs': 16, 'output': 16, 'lhs_c_type': 'bfloat16', 'rhs_c_type': 'bfloat16'},
        {'lhs': 32, 'rhs': 32, 'output': 32, 'lhs_c_type': 'float', 'rhs_c_type': 'float'},
    )
    supported_input_modes = ('direct', 'memtile', 'plio', 'auto')
    supported_output_modes = ('direct', 'memtile', 'plio', 'auto')

    def build_template_params(self, node, config: AddConfig):
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        lhs_view = config.io_views[lhs_tensor.name]
        params = {f: getattr(config, f) for f in config.__dataclass_fields__}
        params.update(
            tile_elements=int(math.prod(lhs_view.tile)),
        )
        return params

    def describe_input_staging(self, node, config, tensor_name, port, buf_dims=None, producer=None):
        if config.preserved_staging is not None:
            return dict(config.preserved_staging[int(port)])
        return describe_elementwise_staging(
            config.io_views[tensor_name], port, 'read', config.staging_contract, buf_dims
        )

    def describe_output_staging(self, node, config, tensor_name, port, buf_dims=None):
        if config.preserved_staging is not None:
            return dict(config.preserved_staging[int(port)])
        return describe_elementwise_staging(
            config.io_views[tensor_name], port, 'write', config.staging_contract, buf_dims
        )

    def output_staging_contract(self, node, config: AddConfig, tensor_name: str):
        return str(config.staging_contract)

    def validate_config(self, node: OpNode, config: AddConfig, device) -> None:
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        lhs_view = config.io_views[lhs_tensor.name]
        if config.preserved_staging is not None:
            if len(config.preserved_staging) != self.output_port_count(node, config):
                raise ValueError(
                    f'{node.name}: preserved_staging length {len(config.preserved_staging)} '
                    f'does not match output_port_count {self.output_port_count(node, config)}.'
                )
        if any(int(dim) <= 0 for dim in lhs_view.tile):
            raise ValueError(f'{node.name}: elementwise Add requires strictly positive slice dimensions.')
        validate_elementwise_tile_contract(
            node_name=node.name,
            precision=config.precision,
            lhs_view=lhs_view,
            bank_bytes=int(device.bank_mem_bytes),
            vec_size=config.vec_size,
        )

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        return {}

    def get_artifacts(self, inst: OpImplInstance):
        return []

    def footprint(self, node: OpNode, config: AddConfig) -> OpImplFootprint:
        return OpImplFootprint(width=1, height=config.parallelism.cas_num, extras={'keepout_left': 1})

    def build_ports(self, node: OpNode, config: AddConfig):
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        return super().build_ports(
            node,
            {lhs_tensor.name: int(config.parallelism.cas_num), rhs_tensor.name: int(config.parallelism.cas_num)},
            int(config.parallelism.cas_num),
        )
