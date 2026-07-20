from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Tuple

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
)
from .common import (
    MICROTILE_OPTIONS,
    bitwidths_supported,
    describe_inner_lhs_staging,
    describe_inner_output_staging,
    describe_outer_lhs_staging,
    describe_outer_output_staging,
    np_bias_dtype_for_spec,
    np_dtype_for_spec,
    pack_as_float,
    pack_mmul_rhs_matrix,
    pack_vector_by_n_slice,
    quantize_to_int,
    requested_contract,
    select_generation_key,
)
from .config import DenseConfig, DenseFlags
from .resolver import (
    _build_matmul_io_views,
    _resolve_bias_dtype,
    _resolve_numeric,
    _resolve_output_scale_shift,
    _resolve_parallelism,
    _resolve_tile_cfg,
)


class _BaseDenseMatmulVariant(OpImplVariant):
    """Unregistered shared base for Dense and Matmul variants."""

    contract: ClassVar[str]

    def build_template_params(self, node, config):
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        lhs_view = config.io_views[lhs_tensor.name]
        output_view = config.io_views[node.outputs[0].name]
        params = {f: getattr(config, f) for f in config.__dataclass_fields__}
        params.update(
            full_outer=self.kernel_outer_extent(lhs_view),
            full_inner_lhs=lhs_view.full_inner,
            full_inner_rhs=output_view.full_inner,
            tile_inner_lhs=lhs_view.tile_inner,
            tile_inner_rhs=output_view.tile_inner,
            tile_inner_lhs_raw=lhs_view.tile_raw_inner,
            tile_inner_rhs_raw=output_view.tile_raw_inner,
        )
        return params

    def kernel_outer_extent(self, lhs_view):
        """Rows this kernel loops over. Contract-specific: declared by each variant."""
        raise NotImplementedError

    def lhs_port_count(self, config) -> int:
        """LHS ports this variant needs. Contract-specific: declared by each variant."""
        raise NotImplementedError

    def microtiling_options(self, generation: str, query) -> List[Tuple[int, int, int]]:
        return list(MICROTILE_OPTIONS.get(select_generation_key(generation), {}).get(tuple(query), []))

    def output_staging_contract(self, _node, config, _tensor_name: str):
        return str(config.parallelism.contract)


class _DenseVariantBase(_BaseDenseMatmulVariant):
    """Everything the dense variants share, independent of which contract they implement."""

    op_type = 'dense'
    graph_header = 'dense_bias_relu_graph.h'
    graph_name = 'dense_bias_relu_graph'
    param_template = 'dense_bias_relu'
    plevel = 10

    def matches(self, node: OpNode, device) -> bool:
        return requested_contract(node) == self.contract and bitwidths_supported(node, device)

    def resolve(self, node: OpNode, device, directives=None) -> DenseConfig:
        io_route, _, _ = parse_directives(directives)
        precision = _resolve_numeric(node, device)
        precision['bias'] = _resolve_bias_dtype(node, precision)
        microtiling = _resolve_tile_cfg(node, device, precision['lhs'], precision['rhs'])
        tiling = _resolve_parallelism(node, device, microtiling, precision, self.contract)
        io_views = _build_matmul_io_views(node, microtiling, tiling)

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
            parallelism=ParallelismConfig(
                cas_length=tiling.cas_length, cas_num=tiling.cas_num, contract=tiling.contract
            ),
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

    def _quantize_weight_bias(self, inst: OpImplInstance):
        """Quantize the weight matrix and bias vector; shared by every dense contract."""
        input_tensor = inst.node.inputs[0]
        weight_tensor = inst.node.inputs[1]
        bias_tensor = inst.node.inputs[2] if len(inst.node.inputs) > 2 else None

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
        return W, b, int(W.shape[-2]), int(W.shape[-1])

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
        n_in = self.lhs_port_count(config)
        n_out = int(config.parallelism.cas_num)
        data_inputs = [t for t in node.inputs if not t.is_parameter]
        return PortMap(
            inputs={t.name: PortBinding(group=f'in{i + 1}', count=n_in) for i, t in enumerate(data_inputs)},
            outputs={t.name: PortBinding(group=f'out{i + 1}', count=n_out) for i, t in enumerate(node.outputs)},
        )


@register_variant
class DenseOpImplVariant(_DenseVariantBase):
    """Dense with the output features partitioned across cascade chains ('inner' contract)."""

    variant_id = 'dense.b.r.v1'
    contract = 'inner'

    def kernel_outer_extent(self, lhs_view):
        return lhs_view.compacted_full_outer

    def lhs_port_count(self, config: DenseConfig) -> int:
        # One LHS slice per cascade column, multicast across every chain.
        return int(config.parallelism.cas_length)

    def describe_input_staging(self, _node, config, tensor_name, port, buf_dims=None, _producer=None):
        return describe_inner_lhs_staging(config.io_views[tensor_name], config.microtiling, port, buf_dims)

    def describe_output_staging(self, _node, config, tensor_name, port, buf_dims=None):
        return describe_inner_output_staging(config.io_views[tensor_name], config.microtiling, port, buf_dims)

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        # 'inner': cas_num slices the columns, so chain c owns weight/bias columns
        # [c*N_slice, (c+1)*N_slice) -- which is exactly what the packers lay out.
        p = inst.config
        W, b, n_in, n_out = self._quantize_weight_bias(inst)
        lhs_view = p.io_views[inst.node.inputs[0].name]
        output_view = p.io_views[inst.node.outputs[0].name]

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


@register_variant
class DenseRowWiseOpImplVariant(_DenseVariantBase):
    """Dense with the rows partitioned across cascade chains ('outer' contract)."""

    variant_id = 'dense.b.r.row.v1'
    contract = 'outer'
    plevel = 10

    def kernel_outer_extent(self, lhs_view):
        return lhs_view.compacted_tile_outer

    def lhs_port_count(self, config: DenseConfig) -> int:
        # Every (chain, column) tile reads its own row slice, so the LHS port array is per-tile.
        return int(config.parallelism.cas_length) * int(config.parallelism.cas_num)

    def describe_input_staging(self, _node, config, tensor_name, port, buf_dims=None, _producer=None):
        return describe_outer_lhs_staging(
            config.io_views[tensor_name], config.microtiling, config.parallelism, port, buf_dims
        )

    def describe_output_staging(self, _node, config, tensor_name, port, buf_dims=None):
        return describe_outer_output_staging(config.io_views[tensor_name], config.microtiling, port, buf_dims)

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        # The packers slice columns as chain*N_slice; here N_slice is the whole N, so pack a
        # single chain (offset 0, full width) and give every row-group that same copy.
        p = inst.config
        W, b, n_in, n_out = self._quantize_weight_bias(inst)
        lhs_view = p.io_views[inst.node.inputs[0].name]
        output_view = p.io_views[inst.node.outputs[0].name]
        cas_num = int(p.parallelism.cas_num)

        packed_W = pack_mmul_rhs_matrix(
            W,
            K=n_in,
            N=n_out,
            K_slice=lhs_view.tile_inner,
            N_slice=output_view.tile_inner,
            microtile_k=p.microtiling.microtile_k,
            microtile_n=p.microtiling.microtile_n,
            cas_length=p.parallelism.cas_length,
            cas_num=1,
            dtype=np_dtype_for_spec(p.precision['rhs']),
        )
        packed_W = np.repeat(packed_W, cas_num, axis=0)

        packed_B = None
        if b is not None:
            packed_B = pack_vector_by_n_slice(
                b,
                N=n_out,
                N_slice=output_view.tile_inner,
                cas_num=1,
                dtype=np_bias_dtype_for_spec(p.precision['bias']),
            )
            packed_B = np.repeat(packed_B, cas_num, axis=0)
        return {'packed_weights': packed_W, 'packed_bias': packed_B}
