# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Shape table built from ONNX shape inference."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .utils import require_onnx


def _static_shape(value_info) -> Optional[Tuple[int, ...]]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField('shape'):
        return None
    dims = tensor_type.shape.dim
    if any(not dim.HasField('dim_value') for dim in dims):
        return None
    return tuple(int(dim.dim_value) for dim in dims)


def normalize_axis(axis: int, rank: int, node_name: str, op_type: str) -> int:
    normalized = axis + rank if axis < 0 else axis
    if not 0 <= normalized < rank:
        raise ValueError(f'{node_name}: {op_type} axis {axis} is out of range for rank {rank}.')
    return normalized


def build_shape_table(
    model_proto,
    initializers: Dict[str, np.ndarray],
    input_shapes: Dict[str, Tuple[int, ...]],
) -> Dict[str, Tuple[int, ...]]:
    """Return every tensor's static shape, keyed by ONNX value name."""
    onnx, _, _ = require_onnx()
    inferred = onnx.shape_inference.infer_shapes(model_proto)

    shape_of: Dict[str, Tuple[int, ...]] = {
        name: tuple(int(x) for x in np.asarray(arr).shape) for name, arr in initializers.items()
    }
    shape_of.update(input_shapes)

    for value_info in list(inferred.graph.value_info) + list(inferred.graph.output):
        shape = _static_shape(value_info)
        if shape is not None:
            shape_of[value_info.name] = shape

    return shape_of
