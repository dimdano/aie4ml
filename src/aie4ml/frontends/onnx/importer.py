# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""ONNX -> aie4ml logical IR importer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ...ir import LogicalIR, TensorVar, set_input_roles
from ...ir.context import AIEBackendContext
from ...model import AIEModel
from . import handlers  # noqa: F401 - registers all handlers
from .context import OnnxImportContext
from .registry import dispatch
from .shapes import build_shape_table
from .utils import (
    create_context,
    initializer_map,
    input_maps,
    require_onnx,
    resolve_project_name,
)
from .utils import (
    node_name as onnx_node_name,
)


def lower_onnx_model(
    model_or_path,
    config: Dict[str, Any],
    *,
    output_dir,
    project_name: Optional[str] = None,
    stamp: Optional[str] = None,
    custom_sources: Optional[Dict[str, str]] = None,
) -> AIEBackendContext:
    onnx, _helper, numpy_helper = require_onnx()

    if isinstance(model_or_path, (str, Path)):
        model_path = Path(model_or_path)
        model_proto = onnx.load(str(model_path))
    else:
        model_path = None
        model_proto = model_or_path
    graph_proto = model_proto.graph

    resolved_project_name = resolve_project_name(model_path, model_proto, project_name)
    backend = create_context(config, output_dir, resolved_project_name, stamp, custom_sources)
    graph: LogicalIR = backend.ir.logical

    initializers = initializer_map(graph_proto, numpy_helper)
    input_shapes, input_dtypes = input_maps(graph_proto, set(initializers))
    for name, shape in input_shapes.items():
        if not shape:
            raise ValueError(f'{name}: scalar inputs are not supported.')
        graph.add_tensor(TensorVar(name=name, shape=tuple(int(x) for x in shape), precision=None))
        graph.mark_graph_input(name)

    ctx = OnnxImportContext(
        backend,
        initializers=initializers,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        shape_of=build_shape_table(model_proto, initializers, input_shapes),
        layer_directives=dict(config.get('LayerDirectives', {}) or {}),
    )

    for index, node in enumerate(graph_proto.node):
        name = onnx_node_name(node, index)
        dispatch(ctx, node, name, ctx.take_directives(name))

    _finalize(ctx, graph_proto)
    return backend


def _finalize(ctx: OnnxImportContext, graph_proto) -> None:
    graph = ctx.graph

    unused = sorted(set(ctx.layer_directives) - ctx.used_directives)
    if unused:
        raise ValueError('Unused LayerDirectives entries: ' + ', '.join(unused))

    for output in graph_proto.output:
        if output.name not in ctx.value_tensors:
            raise ValueError(f'Graph output {output.name}: expected a lowered semantic tensor.')
        tensor = ctx.value_tensors[output.name]
        if output.name not in graph.tensors:
            graph.tensors.pop(tensor.name, None)
            tensor.name = output.name
            graph.tensors[output.name] = tensor
        graph.mark_graph_output(output.name)

    for node in graph.nodes:
        role_names = list(node.metadata.get('input_roles') or [])
        if role_names:
            set_input_roles(node, node.inputs, role_names)

    for tensor in graph.tensors.values():
        if tensor.is_parameter:
            continue
        if tensor.precision is None:
            raise ValueError(f'{tensor.name}: missing quantization intent after ONNX QDQ lowering.')


def from_onnx(
    model_or_path,
    config: Dict[str, Any],
    *,
    output_dir,
    project_name: Optional[str] = None,
    stamp: Optional[str] = None,
    custom_sources: Optional[Dict[str, str]] = None,
) -> AIEModel:
    ctx = lower_onnx_model(
        model_or_path,
        config,
        output_dir=output_dir,
        project_name=project_name,
        stamp=stamp,
        custom_sources=custom_sources,
    )
    return AIEModel.from_context(ctx, source_model=model_or_path)
