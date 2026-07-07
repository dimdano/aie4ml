# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""ONNX op-handler registry."""

from __future__ import annotations

from typing import Callable, Dict

from .context import OnnxImportContext

OnnxHandler = Callable[[OnnxImportContext, object, str, dict], None]

ONNX_HANDLERS: Dict[str, OnnxHandler] = {}


def onnx_handler(*op_types: str) -> Callable[[OnnxHandler], OnnxHandler]:
    """Register `fn` as the handler for one or more ONNX op types."""

    def register(fn: OnnxHandler) -> OnnxHandler:
        for op_type in op_types:
            if op_type in ONNX_HANDLERS:
                raise ValueError(f'duplicate ONNX handler for {op_type!r}: {ONNX_HANDLERS[op_type]} vs {fn}')
            ONNX_HANDLERS[op_type] = fn
        return fn

    return register


def dispatch(ctx: OnnxImportContext, node, node_name: str, directives: dict) -> None:
    handler = ONNX_HANDLERS.get(node.op_type)
    if handler is None:
        raise ValueError(f'{node_name}: unsupported ONNX op {node.op_type}.')
    handler(ctx, node, node_name, directives)
