from __future__ import annotations

import math
from typing import Any, Dict

from ....aie_types import FloatIntent
from ....ir.graph import OpImplInstance, OpNode, input_tensor_for_role
from ...base import OpImplFootprint, OpImplVariant
from ...common_types import PortBinding, PortMap
from ...registry import register_variant
from ...utils import (
    ParallelismConfig,
    align_up,
    build_io_views,
    build_tensor_view_from_staging,
    ceildiv,
    extract_inner_outer,
    find_tile_split,
    parse_directives,
)
from ...utils.io import resolve_input_contract, view_shape
from ...utils.precision import (
    aie_rounding_token,
    infer_accumulator_tag,
    resolve_accumulator_output_shift,
    resolve_exact_storage_dtype,
    storage_bytes_for_spec,
)
from .common import describe_elementwise_staging, elementwise_vec_size
from .config import AddConfig


def _select_preserved_staging(tensor_names, input_contracts):
    primary_name = next((n for n in tensor_names if n in input_contracts), None)
    if primary_name is None:
        return None, {}
    primary = input_contracts[primary_name].port_staging
    patches = {
        n: 'memtile' for n in tensor_names if n in input_contracts and input_contracts[n].port_staging != primary
    }
    return primary, patches


@register_variant
class AddOpImplVariant(OpImplVariant):
    variant_id = 'add.v1'
    op_type = 'add'
    graph_header = 'elementwise_add_graph.h'
    graph_name = 'elementwise_add_graph'
    param_template = 'elementwise_add'
    plevel = 10

    def matches(self, _node: OpNode, device) -> bool:
        return device.generation in ('AIE-ML', 'AIE-MLV2')

    def resolve(self, node: OpNode, device, directives=None) -> AddConfig:
        io_route, input_contracts, parallel_cfg = parse_directives(directives)

        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')

        staging_contract, conflict_patches = resolve_input_contract(
            input_contracts,
            [lhs_tensor.name, rhs_tensor.name],
        )
        preserved_staging, staging_patches = _select_preserved_staging(
            (lhs_tensor.name, rhs_tensor.name),
            input_contracts,
        )
        route_patches = {**conflict_patches, **staging_patches}
        if route_patches:
            io_route = {**io_route, 'inputs': {**io_route.get('inputs', {}), **route_patches}}

        lhs_shape = tuple(int(x) for x in view_shape(node, lhs_tensor, 'inputs'))

        precision = {
            'lhs': resolve_exact_storage_dtype(lhs_tensor.precision, namespace='lhs', layer_name=node.name),
            'rhs': resolve_exact_storage_dtype(rhs_tensor.precision, namespace='rhs', layer_name=node.name),
            'output': resolve_exact_storage_dtype(node.outputs[0].precision, namespace='output', layer_name=node.name),
        }

        is_float = isinstance(lhs_tensor.precision, FloatIntent)
        vec_size = elementwise_vec_size(precision['lhs'], device)
        bank_bytes = int(device.bank_mem_bytes)
        max_rows = max(1, int(device.rows))
        elem_bytes = storage_bytes_for_spec(precision['lhs'])

        if preserved_staging is not None:
            port0 = preserved_staging[0]
            cas_num = len(preserved_staging)
            io_views = {t.name: build_tensor_view_from_staging(node, t, 'inputs', port0) for t in node.inputs}
            io_views.update({t.name: build_tensor_view_from_staging(node, t, 'outputs', port0) for t in node.outputs})
        elif staging_contract == 'inner':
            full_inner, outer_prefix, last_outer = extract_inner_outer(lhs_shape)
            full_inner = align_up(full_inner, vec_size)
            raw_inner = int(lhs_shape[-1])
            compacted_outer = outer_prefix * last_outer
            cas_num, tile_inner = find_tile_split(
                partition_size=full_inner,
                max_rows=max_rows,
                bank_bytes=bank_bytes,
                tile_bytes_fn=lambda ti: compacted_outer * ti * elem_bytes,
                parallel_cfg=parallel_cfg,
                input_contracts=input_contracts,
                primary_tensor_name=lhs_tensor.name,
                contract='inner',
                require_match=True,
            )
            io_views = build_io_views(
                node,
                list(node.inputs),
                list(node.outputs),
                full_inner=full_inner,
                full_outer=last_outer,
                tile_inner=tile_inner,
                tile_outer=last_outer,
                tile_inner_raw=ceildiv(raw_inner, cas_num),
                tile_outer_raw=last_outer,
            )
        else:
            full_inner, outer_prefix, last_outer = extract_inner_outer(lhs_shape)
            full_inner = align_up(full_inner, vec_size)
            raw_inner = int(lhs_shape[-1])
            cas_num, tile_outer = find_tile_split(
                partition_size=last_outer,
                max_rows=max_rows,
                bank_bytes=bank_bytes,
                tile_bytes_fn=lambda to: outer_prefix * to * full_inner * elem_bytes,
                parallel_cfg=parallel_cfg,
                input_contracts=input_contracts,
                primary_tensor_name=lhs_tensor.name,
                contract='outer',
            )
            io_views = build_io_views(
                node,
                list(node.inputs),
                list(node.outputs),
                full_inner=full_inner,
                full_outer=tile_outer * cas_num,
                tile_inner=full_inner,
                tile_outer=tile_outer,
                tile_inner_raw=raw_inner,
                tile_outer_raw=tile_outer,
            )

        if is_float:
            shift, accumulator_tag, rounding_mode = 0, 'accfloat', 'conv_even'
        else:
            shift = resolve_accumulator_output_shift(lhs_tensor.precision, node.outputs[0].precision)
            accumulator_tag = infer_accumulator_tag(device, precision['lhs'], precision['rhs'], precision.get('acc'))
            rounding_mode = aie_rounding_token(precision['output'])

        return AddConfig(
            precision=precision,
            parallelism=ParallelismConfig(cas_num=int(cas_num)),
            vec_size=vec_size,
            io_views=io_views,
            io_route=io_route,
            shift=shift,
            accumulator_tag=accumulator_tag,
            rounding_mode=rounding_mode,
            staging_contract=staging_contract,
            preserved_staging=preserved_staging,
        )

    def validate_config(self, node: OpNode, config: AddConfig, _device) -> None:
        if config.preserved_staging is not None:
            port_count = self.output_port_count(node, config)
            if len(config.preserved_staging) != port_count:
                raise ValueError(
                    f'{node.name}: preserved_staging length {len(config.preserved_staging)} '
                    f'does not match output_port_count {port_count}.'
                )

    def build_template_params(self, node: OpNode, config: AddConfig):
        lhs_view = config.io_views[input_tensor_for_role(node, 'lhs').name]
        params = {f: getattr(config, f) for f in config.__dataclass_fields__}
        params.update(tile_elements=int(math.prod(lhs_view.tile)))
        return params

    def describe_input_staging(self, _node, config, tensor_name, port, buf_dims=None, _producer=None):
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

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        return {}

    def get_artifacts(self, inst: OpImplInstance):
        return []

    def footprint(self, node: OpNode, config: AddConfig) -> OpImplFootprint:
        return OpImplFootprint(width=1, height=config.parallelism.cas_num, extras={'keepout_left': 1})

    def build_ports(self, node: OpNode, config: AddConfig):
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        n = int(config.parallelism.cas_num)
        return PortMap(
            inputs={
                lhs_tensor.name: PortBinding(group='in1', count=n),
                rhs_tensor.name: PortBinding(group='in2', count=n),
            },
            outputs={node.outputs[0].name: PortBinding(group='out1', count=n)},
        )
