# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Kernel metadata registry describing macro-implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .ir.graph import KernelInstance, OpNode, ResolvedAttributes


@dataclass(frozen=True)
class KernelConfig:
    variant_id: str
    param_template: str
    graph_header: str
    graph_name: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'variant_id': self.variant_id,
            'param_template': self.param_template,
            'graph_header': self.graph_header,
            'graph_name': self.graph_name,
            'parameters': dict(self.parameters),
        }


@dataclass(frozen=True)
class KernelSelectionContext:
    node: OpNode
    attributes: ResolvedAttributes
    device_generation: str
    quant: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class KernelPlacementContext:
    node: OpNode
    attributes: ResolvedAttributes
    metadata: Dict[str, Any]
    config: Dict[str, Any]


@dataclass(frozen=True)
class KernelFootprint:
    width: int
    height: int


@dataclass
class KernelVariant:
    variant_id: str
    op_type: str
    supported_generations: Tuple[str, ...] = field(default_factory=tuple)
    supported_precisions: Tuple[Dict[str, int], ...] = field(default_factory=tuple)
    supported_input_modes: Tuple[str, ...] = field(default_factory=tuple)
    supported_output_modes: Tuple[str, ...] = field(default_factory=tuple)
    input_ports: Tuple[str, ...] = field(default_factory=tuple)
    output_ports: Tuple[str, ...] = field(default_factory=tuple)

    def supports_generation(self, generation: str) -> bool:
        if not self.supported_generations:
            return True
        norm = (generation or '').upper()
        for token in self.supported_generations:
            if token.upper() in norm:
                return True
        return False

    def supports(self, context: KernelSelectionContext) -> bool:
        if not self.supports_generation(context.device_generation):
            return False

        if self.supported_input_modes:
            inputs = context.attributes.io_route.get('inputs', {})
            for mode in inputs.values():
                if isinstance(mode, str) and mode not in self.supported_input_modes:
                    return False

        if self.supported_output_modes:
            outputs = context.attributes.io_route.get('outputs', {})
            for mode in outputs.values():
                if isinstance(mode, str) and mode not in self.supported_output_modes:
                    return False

        node_prec = _numeric_precisions(context.attributes)
        if self.supported_precisions:
            if node_prec not in self.supported_precisions:
                return False

        return True

    def build_config(self, context: KernelSelectionContext) -> KernelConfig:
        raise NotImplementedError

    def tiling_options(self, generation: str, input_bits: int, weight_bits: int):
        raise NotImplementedError

    def pack(self, node, attrs, quant_weights, quant_bias, **kwargs):
        raise NotImplementedError

    def describe_output_staging(
        self,
        node: OpNode,
        attrs: ResolvedAttributes,
        tensor_name: str,
        port: int,
        buf_dims=None,
        staging=None,
    ) -> Optional[Dict[str, Any]]:
        return None

    def describe_input_staging(
        self,
        consumer: OpNode,
        attrs: ResolvedAttributes,
        tensor_name: str,
        port: int,
        buf_dims=None,
        staging=None,
        producer: Optional[OpNode] = None,
    ) -> Optional[Dict[str, Any]]:
        return None

    def footprint(self, context: KernelPlacementContext) -> KernelFootprint:
        raise NotImplementedError

    def _build_port_map(
        self,
        context: KernelSelectionContext,
        cas_length: int,
        cas_num: int,
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        ports = {'inputs': {}, 'outputs': {}}

        for i, t in enumerate(context.node.inputs):
            ports['inputs'][t.name] = {'group': f'in{i+1}', 'count': cas_length}

        for i, t in enumerate(context.node.outputs):
            ports['outputs'][t.name] = {'group': f'out{i+1}', 'count': cas_num}

        return ports


def _np_weight_dtype(bitwidth: int):
    return np.int8 if int(bitwidth) <= 8 else np.int16


def _np_bias_dtype(bitwidth: int):
    bw = int(bitwidth)
    return np.int16 if bw <= 16 else np.int32


TILING_OPTIONS: Dict[str, Dict[Tuple[int, int], List[Tuple[int, int, int]]]] = {
    'AIE': {
        (8, 8): [(2, 8, 8), (2, 16, 8), (4, 8, 4), (4, 8, 8), (4, 16, 4), (4, 16, 8), (8, 8, 4)],
        (16, 8): [(4, 4, 4), (4, 4, 8), (4, 8, 4), (8, 4, 4)],
        (8, 16): [(4, 4, 8), (4, 4, 4), (8, 8, 1)],
        (16, 16): [(4, 4, 8), (2, 4, 8), (4, 2, 8), (4, 4, 4), (8, 8, 1)],
    },
    'AIE-ML': {
        (8, 8): [(4, 8, 8), (2, 8, 8), (2, 16, 8), (4, 8, 4), (4, 16, 4), (4, 16, 8), (8, 8, 4), (8, 8, 8)],
        (16, 8): [(4, 4, 8), (2, 8, 8), (4, 4, 4), (4, 8, 4), (8, 4, 4), (8, 4, 8)],
        (8, 16): [(4, 4, 4), (4, 4, 8)],
        (16, 16): [(4, 4, 4), (2, 4, 8), (4, 2, 8), (4, 4, 8), (8, 1, 8), (8, 2, 8)],
    },
    'AIE-MLV2': {
        (8, 8): [(8, 8, 8), (4, 8, 8)],
        (16, 8): [(4, 4, 8), (8, 2, 8)],
        (8, 16): [(4, 4, 8), (8, 2, 8)],
        (16, 16): [(8, 2, 8)],
    },
}


def _select_generation_key(generation: str) -> str:
    norm = (generation or '').upper()
    for key in sorted(TILING_OPTIONS.keys(), key=len, reverse=True):
        if key in norm:
            return key
    return 'AIE'


def _numeric_precisions(attrs: ResolvedAttributes) -> Dict[str, int]:
    out = {}
    for kind, dtype in attrs.numeric.items():
        out[kind] = int(dtype.width)
    return out


def _serialize_dtype(dtype) -> Dict[str, Any]:
    if dtype is None:
        raise RuntimeError('Kernel precision missing numeric spec.')
    return {
        'width': int(dtype.width),
        'signed': bool(dtype.signed),
        'frac': int(dtype.frac),
        'c_type': dtype.c_type,
    }


@dataclass
class DenseKernelVariant(KernelVariant):
    staging_scheme_in: str = 'dense_staging_ifm'
    staging_scheme_out: str = 'dense_staging_ofm'

    def _io_view(self, node: OpNode, tensor_name: str, direction: str) -> Dict[str, Any]:
        return node.traits['io_view'].data[direction][tensor_name]

    def _io_shapes(self, attrs: ResolvedAttributes, tensor_name: str, direction: str) -> Dict[str, List[int]]:
        return attrs.scalars['io_shapes'][direction][tensor_name]

    def default_input_staging(self, node, tensor_name):
        return {'scheme': self.staging_scheme_in}

    def default_output_staging(self, node, tensor_name):
        return {'scheme': self.staging_scheme_out}

    def describe_input_staging(
        self,
        node: OpNode,
        attrs: ResolvedAttributes,
        tensor_name: str,
        port: int,
        buf_dims=None,
        staging=None,
        producer=None,
    ) -> Dict[str, Any]:
        scheme = self.staging_scheme_in if staging is None else staging['scheme']
        if scheme == 'dense_staging_ifm':
            return self._describe_dense_ifm(node, attrs, tensor_name, port, buf_dims)

        raise ValueError(f"{self.variant_id}: unsupported input staging scheme '{scheme}'")

    def describe_output_staging(
        self,
        node: OpNode,
        attrs: ResolvedAttributes,
        tensor_name: str,
        port: int,
        buf_dims=None,
        staging=None,
    ) -> Dict[str, Any]:
        scheme = self.staging_scheme_out if staging is None else staging['scheme']
        if scheme == 'dense_staging_ofm':
            return self._describe_dense_ofm(node, attrs, tensor_name, port, buf_dims)

        raise ValueError(f"{self.variant_id}: unsupported output staging scheme '{scheme}'")

    def supports(self, context: KernelSelectionContext) -> bool:
        if not super().supports(context):
            return False

        attrs = context.attributes
        tile_m = int(attrs.tiling['tile_m'])
        tile_k = int(attrs.tiling['tile_k'])
        tile_n = int(attrs.tiling['tile_n'])
        input_bits = int(attrs.numeric['input'].width)
        weight_bits = int(attrs.numeric['weight'].width)

        if not all((tile_m, tile_k, tile_n, input_bits, weight_bits)):
            return False

        options = self.tiling_options(context.device_generation, input_bits, weight_bits)
        return (tile_m, tile_k, tile_n) in options

    def build_config(self, context: KernelSelectionContext) -> KernelConfig:
        attrs = context.attributes
        parallel = dict(attrs.parallelism)
        tiling = dict(attrs.tiling)
        slices = dict(attrs.slices)
        scalars = dict(attrs.scalars)
        flags = dict(attrs.flags)
        cas = int(parallel['cas_length'])
        chains = int(parallel['cas_num'])

        precision = {
            key: _serialize_dtype(attrs.numeric.get(key))
            for key in ('input', 'weight', 'output', 'bias', 'acc')
            if attrs.numeric.get(key) is not None
        }

        # TODO: Duplicate of data already present in ResolvedAttributes; should
        # pass attrs directly into template context to avoid double-sourcing.
        return KernelConfig(
            variant_id=self.variant_id,
            param_template='dense_bias_relu',
            graph_header='dense_bias_relu_graph.h',
            graph_name='dense_bias_relu_graph',
            parameters={
                'precision': precision,
                'parallelism': {
                    'cas_length': cas,
                    'cas_num': chains,
                    'input_slice_raw': int(slices['input_raw']),
                    'output_slice_raw': int(slices['output_raw']),
                },
                'tiling': tiling,
                'flags': flags,
                'in_feat_slice': int(slices['input']),
                'out_feat_slice': int(slices['output']),
                'padded_independent_extent': int(scalars['padded_independent_extent']),
                'padded_in_features': int(scalars['padded_in_features']),
                'padded_out_features': int(scalars['padded_out_features']),
                'shift': int(scalars['shift']),
                'accumulator_tag': scalars.get('accumulator_tag'),
                'rounding_mode': scalars.get('rounding_mode'),
                'ports': self._build_port_map(context, cas, chains),
            },
        )

    def tiling_options(self, generation: str, input_bits: int, weight_bits: int) -> List[Tuple[int, int, int]]:
        gen_key = _select_generation_key(generation)
        bucket = TILING_OPTIONS.get(gen_key, {})
        return list(bucket.get((int(input_bits), int(weight_bits)), []))

    def pack(self, inst: KernelInstance) -> Dict[str, Any]:
        attrs = inst.attributes
        parallel = attrs.parallelism
        tiling = attrs.tiling
        slices = attrs.slices

        W = inst.node.artifacts['quant_weights']
        b = inst.node.artifacts.get('quant_bias')

        W = np.asarray(W)
        if W.ndim < 2:
            raise ValueError(f'{inst.name}: quant_weights must have at least 2 dimensions, got {W.ndim}.')
        n_in = int(W.shape[-2])
        n_out = int(W.shape[-1])

        weight_spec = attrs.numeric['weight']
        bias_spec = attrs.numeric['bias']
        weight_dtype = _np_weight_dtype(int(weight_spec.width))
        bias_dtype = _np_bias_dtype(int(bias_spec.width))

        from .passes.pack import pack_mmul_rhs_matrix, pack_vector_by_n_slice

        packed_W = pack_mmul_rhs_matrix(
            W,
            K=n_in,
            N=n_out,
            K_slice=slices['input'],
            N_slice=slices['output'],
            tile_k=tiling['tile_k'],
            tile_n=tiling['tile_n'],
            cas_length=parallel['cas_length'],
            cas_num=parallel['cas_num'],
            dtype=weight_dtype,
        )
        packed_B = (
            pack_vector_by_n_slice(
                b,
                N=n_out,
                N_slice=slices['output'],
                cas_num=parallel['cas_num'],
                dtype=bias_dtype,
            )
            if b is not None
            else None
        )

        return {'packed_weights': packed_W, 'packed_bias': packed_B}

    def _describe_dense_ofm(
        self,
        node: OpNode,
        attrs: ResolvedAttributes,
        tensor_name: str,
        port: int,
        buf_dims=None,
    ) -> Dict[str, Any]:
        """
        Dense output tensor:
          dims = [padded_out_features, padded_batch]
          kernel consumes tiles [tile_n, tile_m]
        """
        tile_m = int(attrs.tiling['tile_m'])
        tile_n = int(attrs.tiling['tile_n'])
        out_slice = int(attrs.slices['output'])
        raw_out = int(attrs.slices['output_raw'])
        view = self._io_view(node, tensor_name, 'outputs')
        shapes = self._io_shapes(attrs, tensor_name, 'outputs')

        buffer_order = list(view['buffer_order'])
        buffer_dimension = [int(shapes['padded'][i]) for i in buffer_order]
        io_boundary_dimension = [int(shapes['real'][i]) for i in buffer_order]
        io_tiling_dimension = list(io_boundary_dimension)

        feat_dim = buffer_order.index(int(view['feature_axis']))
        indep_dim = buffer_order.index(int(view['independent_axes'][-1]))

        io_tiling_dimension[feat_dim] = raw_out
        tiling_dimension = [1 for _ in buffer_dimension]
        tiling_dimension[feat_dim] = tile_n
        tiling_dimension[indep_dim] = tile_m

        tile_traversal = []
        tile_traversal.append({'dimension': feat_dim, 'stride': tile_n, 'wrap': out_slice // tile_n})
        tile_traversal.append({'dimension': indep_dim, 'stride': tile_m, 'wrap': buffer_dimension[indep_dim] // tile_m})
        for dim in range(len(buffer_dimension)):
            if dim in (feat_dim, indep_dim):
                continue
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
        }

    def _describe_dense_ifm(
        self,
        consumer: OpNode,
        attrs: ResolvedAttributes,
        tensor_name: str,
        port: int,
        buf_dims=None,
    ) -> Dict[str, Any]:
        """
        Dense input tensor:
          dims = [padded_in_features, padded_batch]
          kernel consumes tiles [tile_k, tile_m]
        """
        tile_m = int(attrs.tiling['tile_m'])
        tile_k = int(attrs.tiling['tile_k'])
        in_slice = int(attrs.slices['input'])
        raw_in = int(attrs.slices['input_raw'])
        view = self._io_view(consumer, tensor_name, 'inputs')
        shapes = self._io_shapes(attrs, tensor_name, 'inputs')

        buffer_order = list(view['buffer_order'])
        buffer_dimension = [int(shapes['padded'][i]) for i in buffer_order]
        boundary_dimension = [int(shapes['real'][i]) for i in buffer_order]
        io_boundary_dimension = list(boundary_dimension)
        io_tiling_dimension = list(io_boundary_dimension)

        feat_dim = buffer_order.index(int(view['feature_axis']))
        indep_dim = buffer_order.index(int(view['independent_axes'][-1]))

        io_tiling_dimension[feat_dim] = raw_in
        tiling_dimension = [1 for _ in buffer_dimension]
        tiling_dimension[feat_dim] = tile_k
        tiling_dimension[indep_dim] = tile_m

        tile_traversal = []
        tile_traversal.append({'dimension': feat_dim, 'stride': tile_k, 'wrap': in_slice // tile_k})
        tile_traversal.append({'dimension': indep_dim, 'stride': tile_m, 'wrap': buffer_dimension[indep_dim] // tile_m})
        for dim in range(len(buffer_dimension)):
            if dim in (feat_dim, indep_dim):
                continue
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
        }

    def footprint(self, context: KernelPlacementContext) -> KernelFootprint:
        par = context.attributes.parallelism
        return KernelFootprint(
            width=int(par['cas_length']),
            height=int(par['cas_num']),
        )

    def get_artifacts(self, inst: KernelInstance):
        return [
            {
                'name': 'weights',
                'kind': '2d',
                'array': inst.artifacts['packed_weights'],
                'dtype': inst.config['parameters']['precision']['weight']['c_type'],
                'filename': f'weights_{inst.name}.h',
                'port': 'wts',
            },
            {
                'name': 'bias',
                'kind': '1d',
                'array': inst.artifacts.get('packed_bias'),
                'dtype': inst.config['parameters']['precision']['bias']['c_type'],
                'filename': f'bias_{inst.name}.h',
                'port': 'bias',
            },
        ]


class KernelRegistry:
    def __init__(self):
        self._variants: Dict[str, List[KernelVariant]] = {}

    def register(self, variant: KernelVariant) -> None:
        self._variants.setdefault(variant.op_type, []).append(variant)

    def variants(self, op_type: str) -> Iterable[KernelVariant]:
        return self._variants.get(op_type, [])

    def select(self, context: KernelSelectionContext) -> Optional[KernelVariant]:
        candidates = self._variants.get(context.node.op_type, [])
        if not candidates:
            return None
        for variant in candidates:
            if variant.supports(context):
                return variant
        return None

    def supported_tilings(
        self, op_type: str, generation: str, input_bits: int, weight_bits: int
    ) -> List[Tuple[int, int, int]]:
        candidates = self._variants.get(op_type, [])
        variant = None
        for cand in candidates:
            if cand.supports_generation(generation):
                variant = cand
                break
        if variant is None and candidates:
            variant = candidates[0]
        if variant is None:
            return []
        return variant.tiling_options(generation, input_bits, weight_bits)


_GLOBAL_REGISTRY = KernelRegistry()
_GLOBAL_REGISTRY.register(
    DenseKernelVariant(
        variant_id='dense.b.r.v1',
        op_type='dense',
        supported_generations=('AIE-ML', 'AIE-MLV2'),
        supported_precisions=(
            {'input': 8, 'weight': 8, 'output': 8, 'acc': 32, 'bias': 32},
            {'input': 8, 'weight': 8, 'output': 16, 'acc': 32, 'bias': 32},
            {'input': 8, 'weight': 8, 'output': 32, 'acc': 32, 'bias': 32},
            {'input': 16, 'weight': 8, 'output': 8, 'acc': 32, 'bias': 32},
            {'input': 16, 'weight': 8, 'output': 8, 'acc': 32, 'bias': 32},
            {'input': 16, 'weight': 16, 'output': 16, 'acc': 64, 'bias': 32},
            {'input': 16, 'weight': 16, 'output': 32, 'acc': 64, 'bias': 32},
        ),
        supported_input_modes=('direct', 'memtile', 'plio'),
        supported_output_modes=('direct', 'memtile', 'plio'),
    )
)


def get_kernel_registry() -> KernelRegistry:
    return _GLOBAL_REGISTRY
