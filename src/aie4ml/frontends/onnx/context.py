# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Import state and tensor plumbing shared by all ONNX handlers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ...aie_types import FloatFormat, FloatIntent, QuantIntent
from ...ir.context import AIEBackendContext
from ...ir.graph import LogicalIR, OpNode, TensorVar
from .utils import intent_from_initializer

_ONNX_FLOAT = 1
_ONNX_BFLOAT16 = 16
_ONNX_FP8E4M3FN = 17

_FLOAT_INPUT_INTENT = {
    _ONNX_FLOAT: FloatIntent(width=32, format=FloatFormat.FP32),
    _ONNX_BFLOAT16: FloatIntent(width=16, format=FloatFormat.BF16),
    _ONNX_FP8E4M3FN: FloatIntent(width=8, format=FloatFormat.FP8_E4M3),
}


class OnnxImportContext:
    """Mutable state threaded through the ONNX handler dispatch."""

    def __init__(
        self,
        backend: AIEBackendContext,
        *,
        initializers: Dict[str, np.ndarray],
        input_shapes: Dict[str, Tuple[int, ...]],
        input_dtypes: Dict[str, int],
        shape_of: Dict[str, Tuple[int, ...]],
        layer_directives: Dict[str, Any],
    ):
        self.backend = backend
        self.graph: LogicalIR = backend.ir.logical
        self.device = backend.device
        self.initializers = initializers
        self.input_shapes = input_shapes
        self.input_dtypes = input_dtypes
        self.shape_of = shape_of
        self.layer_directives = layer_directives
        self.used_directives: set[str] = set()

        # ONNX value name -> lowered activation/parameter tensor.
        self.value_tensors: Dict[str, TensorVar] = {}
        # QuantizeLinear output name -> (kind, ref, intent, shape) pending a DequantizeLinear.
        self.q_aliases: Dict[str, Tuple[str, Any, QuantIntent, Tuple[int, ...]]] = {}

    # -- directives ---------------------------------------------------------

    def take_directives(self, node_name: str) -> Dict[str, Any]:
        from .utils import normalize_directives

        directives = normalize_directives(node_name, self.layer_directives.get(node_name))
        if directives:
            self.used_directives.add(node_name)
        return directives

    def output_shape(self, name: str, node_name: str) -> Tuple[int, ...]:
        if name not in self.shape_of:
            raise ValueError(f'{node_name}: shape inference produced no shape for {name!r}.')
        return tuple(int(x) for x in self.shape_of[name])

    # -- tensor construction ------------------------------------------------

    def graph_input_tensor(self, name: str, shape: Sequence[int], intent: QuantIntent) -> TensorVar:
        tensor = self.graph.tensors.get(name)
        if tensor is None:
            tensor = TensorVar(name=name, shape=tuple(int(x) for x in shape), precision=intent)
            self.graph.add_tensor(tensor)
        elif tensor.precision is None:
            tensor.precision = intent
        elif tensor.precision != intent:
            raise ValueError(f'{name}: conflicting quantization intent for graph input.')
        return tensor

    def param_tensor(self, name: str, data: np.ndarray, intent: QuantIntent) -> TensorVar:
        tensor = self.graph.tensors.get(name)
        shape = tuple(int(x) for x in np.asarray(data).shape)
        if tensor is None:
            tensor = TensorVar(name=name, shape=shape, precision=intent, data=np.asarray(data, dtype=np.float64))
            self.graph.add_tensor(tensor)
            return tensor
        if tensor.data is None:
            raise ValueError(f'{name}: expected parameter tensor.')
        if tensor.precision != intent:
            raise ValueError(f'{name}: conflicting quantization intent for parameter tensor.')
        return tensor

    # -- source lookup ------------------------------------------------------

    def source_for(self, name: str, node_name: str) -> TensorVar:
        """An activation tensor; never a constant."""
        if name in self.value_tensors:
            return self.value_tensors[name]
        if name in self.input_shapes:
            tensor = self.any_source_for(name, node_name)
            if tensor.is_parameter:
                raise ValueError(f'{node_name}: activation input {name} cannot be constant.')
            return tensor
        raise ValueError(f'{node_name}: unsupported input {name}. Expected a dequantized activation tensor.')

    def parameter_source_for(self, name: str, node_name: str) -> TensorVar:
        """A constant tensor, materialized from an initializer if needed."""
        if name in self.value_tensors:
            tensor = self.value_tensors[name]
            if not tensor.is_parameter:
                raise ValueError(f'{node_name}: parameter input {name} must be constant.')
            return tensor
        if name in self.initializers:
            data = np.asarray(self.initializers[name])
            return self.param_tensor(name, data, intent_from_initializer(data, node_name))
        raise ValueError(f'{node_name}: parameter input {name} must be a constant initializer.')

    def any_source_for(self, name: str, node_name: str) -> TensorVar:
        """Either an activation, a parameter, or a bare float graph input."""
        if name in self.value_tensors:
            return self.value_tensors[name]
        if name in self.initializers:
            return self.parameter_source_for(name, node_name)
        if name in self.input_shapes:
            elem_type = self.input_dtypes[name]
            intent = _FLOAT_INPUT_INTENT.get(elem_type)
            if intent is None:
                raise ValueError(
                    f'{node_name}: graph input "{name}" has unsupported ONNX elem_type {elem_type}. '
                    'Use QDQ wrapping for integer inputs.'
                )
            tensor = self.graph_input_tensor(name, self.input_shapes[name], intent)
            self.value_tensors[name] = tensor
            return tensor
        raise ValueError(f'{node_name}: unsupported input {name}.')

    # -- registration / emission -------------------------------------------

    def bind(self, name: str, tensor: TensorVar) -> None:
        self.value_tensors[name] = tensor

    def emit(
        self,
        op_type: str,
        node_name: str,
        *,
        inputs: Sequence[TensorVar],
        outputs: Sequence[Tuple[str, Sequence[int], Optional[Any]]],
        metadata: Dict[str, Any],
        directives: Dict[str, Any],
        roles: Optional[Sequence[str]] = None,
    ) -> OpNode:
        """Create a semantic OpNode, wire it, and register its outputs."""
        op = OpNode(name=f'{node_name}_aie', op_type=op_type, dialect=self.device.dialect)
        op.metadata.update(metadata)
        if roles is not None:
            op.metadata['input_roles'] = list(roles)
        op.directives.update(directives)

        for src in inputs:
            src.consumers.append(op)
        op.inputs.extend(inputs)

        for out_name, out_shape, out_precision in outputs:
            out_tensor = TensorVar(
                name=out_name,
                shape=tuple(int(x) for x in out_shape),
                precision=out_precision,
                producer=op,
            )
            self.graph.add_tensor(out_tensor)
            op.outputs.append(out_tensor)
            self.value_tensors[out_name] = out_tensor

        self.graph.add_node(op)
        return op
