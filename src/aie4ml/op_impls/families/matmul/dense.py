from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ....ir.graph import OpImplInstance, OpNode
from ...base import (
    OpImplConfig,
    OpImplFootprint,
    OpImplPlacementContext,
    OpImplSelectionContext,
    OpImplVariant,
)
from .common import TILING_OPTIONS, np_bias_dtype_for_spec, np_dtype_for_spec, select_generation_key, tiling_key
from .types import MatmulParallelismConfig, MatmulTilingConfig


@dataclass(frozen=True)
class DenseFlags:
    """Dense implementation behaviour flags used by templates and codegen."""

    use_relu: bool
    transpose_input: bool
    use_bias: bool


@dataclass(frozen=True)
class DenseOpImplParameters:
    """Dense-specific compile-time parameters for the selected AIE implementation."""

    precision: Dict[str, Any]
    parallelism: MatmulParallelismConfig
    tiling: MatmulTilingConfig
    flags: DenseFlags
    in_feat_slice: int
    out_feat_slice: int
    input_slice_raw: int
    output_slice_raw: int
    padded_independent_extent: int
    padded_in_features: int
    padded_out_features: int
    shift: int
    accumulator_tag: Optional[str]
    rounding_mode: Optional[str]
    io_shapes: Dict[str, Any]


class DenseOpImplVariant(OpImplVariant):
    def _io_view(self, node: OpNode, tensor_name: str, direction: str) -> Dict[str, Any]:
        return node.traits['io_view'].data[direction][tensor_name]

    def _map_view_axis(self, view: Dict[str, Any], axis: int) -> int:
        perm = view.get('perm')
        if perm is None:
            return int(axis)
        return perm[axis]

    def _canonical_buffer_axes(
        self, view: Dict[str, Any], rank: int, buffer_order: List[int]
    ) -> Tuple[int, int, List[int]]:
        feat_axis = self._map_view_axis(view, rank - 1)
        indep_axis = self._map_view_axis(view, rank - 2)
        feat_dim = buffer_order.index(int(feat_axis))
        indep_dim = buffer_order.index(int(indep_axis))
        tail_dims = sorted(dim for dim in range(rank) if dim not in (feat_dim, indep_dim))
        return feat_dim, indep_dim, [feat_dim, indep_dim] + tail_dims

    def describe_input_staging(self, node, config, tensor_name, port, buf_dims=None, producer=None):
        return self._describe_dense_ifm(node, config, tensor_name, port, buf_dims)

    def describe_output_staging(self, node, config, tensor_name, port, buf_dims=None):
        return self._describe_dense_ofm(node, config, tensor_name, port, buf_dims)

    def supports(self, context: OpImplSelectionContext) -> bool:
        if not super().supports(context):
            return False

        attrs = context.attributes
        tile_m = int(attrs.tiling['tile_m'])
        tile_k = int(attrs.tiling['tile_k'])
        tile_n = int(attrs.tiling['tile_n'])
        input_key = tiling_key(attrs.numeric['input'])
        weight_key = tiling_key(attrs.numeric['weight'])

        if not all((tile_m, tile_k, tile_n, input_key, weight_key)):
            return False
        return (tile_m, tile_k, tile_n) in self.tiling_options(context.device_generation, (input_key, weight_key))

    def build_config(self, context: OpImplSelectionContext) -> OpImplConfig:
        attrs = context.attributes
        parallel = dict(attrs.parallelism)
        tiling = dict(attrs.tiling)
        slices = dict(attrs.slices)
        scalars = dict(attrs.scalars)
        flags = dict(attrs.flags)
        input_view = self._io_view(context.node, context.node.inputs[0].name, 'inputs')
        perm = input_view.get('perm')
        transpose_input = perm is not None and perm[-1] != (len(perm) - 1)
        cas = int(parallel['cas_length'])
        chains = int(parallel['cas_num'])

        params = DenseOpImplParameters(
            precision={
                key: attrs.numeric.get(key)
                for key in ('input', 'weight', 'output', 'bias', 'acc')
                if attrs.numeric.get(key) is not None
            },
            parallelism=MatmulParallelismConfig(
                cas_length=cas,
                cas_num=chains,
            ),
            tiling=MatmulTilingConfig(
                tile_m=int(tiling['tile_m']),
                tile_k=int(tiling['tile_k']),
                tile_n=int(tiling['tile_n']),
            ),
            flags=DenseFlags(
                use_relu=bool(flags['use_relu']),
                transpose_input=bool(transpose_input),
                use_bias=bool(context.node.metadata['use_bias']),
            ),
            in_feat_slice=int(slices['input']),
            out_feat_slice=int(slices['output']),
            input_slice_raw=int(slices['input_raw']),
            output_slice_raw=int(slices['output_raw']),
            padded_independent_extent=int(scalars['padded_independent_extent']),
            padded_in_features=int(scalars['padded_in_features']),
            padded_out_features=int(scalars['padded_out_features']),
            shift=int(scalars['shift']),
            accumulator_tag=scalars.get('accumulator_tag'),
            rounding_mode=scalars.get('rounding_mode'),
            io_shapes=scalars['io_shapes'],
        )

        return OpImplConfig(
            variant_id=self.variant_id,
            param_template='dense_bias_relu',
            graph_header='dense_bias_relu_graph.h',
            graph_name='dense_bias_relu_graph',
            parameters=params,
            ports=self._build_port_map(context, cas, chains),
            io_route=dict(attrs.io_route),
        )

    def tiling_options(self, generation: str, query) -> List[Tuple[int, int, int]]:
        return list(TILING_OPTIONS.get(select_generation_key(generation), {}).get(tuple(query), []))

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        from ....aie_types import FloatFormat, FloatIntent
        from .common import pack_as_float as _pack_as_float
        from .common import pack_mmul_rhs_matrix, pack_vector_by_n_slice
        from .common import quantize_to_int as _quantize_to_int

        p = inst.config.parameters
        input_tensor = inst.node.inputs[0]
        weight_tensor = inst.node.inputs[1]
        bias_tensor = inst.node.inputs[2] if len(inst.node.inputs) > 2 else None

        wi = weight_tensor.precision
        if isinstance(wi, FloatIntent):
            W = _pack_as_float(weight_tensor.data, wi.format)
            b = np.asarray(bias_tensor.data, dtype=np.float32) if bias_tensor is not None else None
        else:
            W = _quantize_to_int(
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
                b = _quantize_to_int(
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
            K_slice=p.in_feat_slice,
            N_slice=p.out_feat_slice,
            tile_k=p.tiling.tile_k,
            tile_n=p.tiling.tile_n,
            cas_length=p.parallelism.cas_length,
            cas_num=p.parallelism.cas_num,
            dtype=np_dtype_for_spec(p.precision['weight']),
        )
        if isinstance(wi, FloatIntent) and wi.format == FloatFormat.BF16:
            packed_W = (packed_W.astype(np.uint32) << 16).view(np.float32)
        packed_B = (
            pack_vector_by_n_slice(
                b,
                N=n_out,
                N_slice=p.out_feat_slice,
                cas_num=p.parallelism.cas_num,
                dtype=np_bias_dtype_for_spec(p.precision['bias']),
            )
            if b is not None
            else None
        )
        return {'packed_weights': packed_W, 'packed_bias': packed_B}

    def _describe_dense_ofm(self, node, config, tensor_name, port, buf_dims=None):
        p = config.parameters
        tile_m = int(p.tiling.tile_m)
        tile_n = int(p.tiling.tile_n)
        out_slice = int(p.out_feat_slice)
        raw_out = int(p.output_slice_raw)
        view = self._io_view(node, tensor_name, 'outputs')
        shapes = p.io_shapes['outputs'][tensor_name]
        buffer_order = list(view['buffer_order'])
        buffer_dimension = (
            [int(shapes['padded'][i]) for i in buffer_order] if buf_dims is None else [int(x) for x in buf_dims]
        )
        io_boundary_dimension = [int(shapes['real'][i]) for i in buffer_order]
        io_tiling_dimension = list(io_boundary_dimension)
        feat_dim, indep_dim, traversal_dims = self._canonical_buffer_axes(view, len(buffer_dimension), buffer_order)
        io_tiling_dimension[feat_dim] = raw_out
        tiling_dimension = [1 for _ in buffer_dimension]
        tiling_dimension[feat_dim] = tile_n
        tiling_dimension[indep_dim] = tile_m
        tile_traversal = []
        for dim in traversal_dims:
            if dim == feat_dim:
                tile_traversal.append({'dimension': feat_dim, 'stride': tile_n, 'wrap': out_slice // tile_n})
            elif dim == indep_dim:
                tile_traversal.append(
                    {'dimension': indep_dim, 'stride': tile_m, 'wrap': buffer_dimension[indep_dim] // tile_m}
                )
            else:
                tile_traversal.append({'dimension': dim, 'stride': 1, 'wrap': buffer_dimension[dim]})
        offset = [0 for _ in buffer_dimension]
        offset[feat_dim] = port * out_slice
        return {
            'access': 'write',
            'buffer_dimension': buffer_dimension,
            'tiling_dimension': tiling_dimension,
            'offset': offset,
            'tile_traversal': tile_traversal,
            'io_tiling_dimension': io_tiling_dimension,
            'io_boundary_dimension': io_boundary_dimension,
            'slice_dimension': feat_dim,
            'feature_dimension': feat_dim,
            'independent_dimension': indep_dim,
        }

    def _describe_dense_ifm(self, consumer, config, tensor_name, port, buf_dims=None):
        p = config.parameters
        tile_m = int(p.tiling.tile_m)
        tile_k = int(p.tiling.tile_k)
        in_slice = int(p.in_feat_slice)
        raw_in = int(p.input_slice_raw)
        view = self._io_view(consumer, tensor_name, 'inputs')
        shapes = p.io_shapes['inputs'][tensor_name]
        buffer_order = list(view['buffer_order'])
        buffer_dimension = (
            [int(shapes['padded'][i]) for i in buffer_order] if buf_dims is None else [int(x) for x in buf_dims]
        )
        boundary_dimension = [int(shapes['logical'][i]) for i in buffer_order]
        io_boundary_dimension = list(boundary_dimension)
        io_tiling_dimension = list(io_boundary_dimension)
        feat_dim, indep_dim, traversal_dims = self._canonical_buffer_axes(view, len(buffer_dimension), buffer_order)
        io_tiling_dimension[feat_dim] = raw_in
        tiling_dimension = [1 for _ in buffer_dimension]
        tiling_dimension[feat_dim] = tile_k
        tiling_dimension[indep_dim] = tile_m
        tile_traversal = []
        for dim in traversal_dims:
            if dim == feat_dim:
                tile_traversal.append({'dimension': feat_dim, 'stride': tile_k, 'wrap': in_slice // tile_k})
            elif dim == indep_dim:
                tile_traversal.append(
                    {'dimension': indep_dim, 'stride': tile_m, 'wrap': buffer_dimension[indep_dim] // tile_m}
                )
            else:
                tile_traversal.append({'dimension': dim, 'stride': 1, 'wrap': buffer_dimension[dim]})
        offset = [0 for _ in buffer_dimension]
        offset[feat_dim] = port * in_slice
        return {
            'access': 'read',
            'buffer_dimension': buffer_dimension,
            'tiling_dimension': tiling_dimension,
            'offset': offset,
            'tile_traversal': tile_traversal,
            'boundary_dimension': boundary_dimension,
            'io_tiling_dimension': io_tiling_dimension,
            'io_boundary_dimension': io_boundary_dimension,
            'slice_dimension': feat_dim,
            'feature_dimension': feat_dim,
            'independent_dimension': indep_dim,
        }

    def footprint(self, context: OpImplPlacementContext) -> OpImplFootprint:
        p = context.config.parameters
        return OpImplFootprint(width=p.parallelism.cas_length, height=p.parallelism.cas_num)

    def get_artifacts(self, inst: OpImplInstance):
        return [
            {
                'name': 'weights',
                'kind': '2d',
                'storage': 'rom',
                'array': inst.artifacts['packed_weights'],
                'dtype': inst.config.parameters.precision['weight'].c_type,
                'filename': f'weights_{inst.name}.h',
                'port': 'wts',
            },
            {
                'name': 'bias',
                'kind': '1d',
                'storage': 'rom',
                'array': inst.artifacts.get('packed_bias'),
                'dtype': inst.config.parameters.precision['bias'].c_type,
                'filename': f'bias_{inst.name}.h',
                'port': 'bias',
            },
        ]
