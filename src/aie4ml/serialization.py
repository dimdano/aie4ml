# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Serialization helpers for exporting the backend IR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .aie_types import QuantIntent
from .ir.graph import KernelInstance, OpNode


def dump_pipeline_ir(ctx, destination: Path) -> None:
    """Serialize logical, kernel, and physical IR into a single JSON file."""

    data = {
        'logical': [serialize_logical_node(node) for node in ctx.ir.logical],
        'kernels': [serialize_kernel_instance(inst) for inst in ctx.ir.kernels],
        'physical': serialize_physical_ir(ctx.ir.physical),
    }

    destination.write_text(json.dumps(data, indent=2))


def serialize_logical_node(node: OpNode) -> Dict[str, Any]:
    metadata = _serialize_metadata(node.metadata)

    return {
        'name': node.name,
        'op_type': node.op_type,
        'dialect': node.dialect,
        'inputs': [t.name for t in node.inputs],
        'outputs': [t.name for t in node.outputs],
        'traits': {name: trait.data for name, trait in node.traits.items()},
        'metadata': metadata,
        # TODO: serialize node.artifacts (external .npy blobs + references)
    }


def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(metadata)
    quant_meta = metadata.get('quant')
    if isinstance(quant_meta, dict):
        metadata['quant'] = _serialize_quant_metadata(quant_meta)

    return metadata


def _serialize_quant_metadata(quant_meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in quant_meta.items():
        if key.endswith('_precision'):
            out[key] = serialize_precision(value)
        else:
            out[key] = value
    return out


def serialize_kernel_instance(inst: KernelInstance) -> Dict[str, Any]:
    return {
        'node': inst.name,
        'variant_id': inst.variant.variant_id,
        'kernel_config': inst.config,
        # "input_staging": inst.variant.describe_input_staging(inst),
        # "output_staging": inst.variant.describe_output_staging(inst),
        # TODO: serialize kernel artifacts (weights, LUTs, etc.)
    }


def serialize_physical_ir(physical_ir) -> Dict[str, Any]:
    return {
        'placements': physical_ir.placements,
        'plan': physical_ir.plan,
    }


def serialize_precision(precision):
    if precision is None:
        return None
    from .aie_types import FloatIntent

    if isinstance(precision, FloatIntent):
        return {'width': int(precision.width), 'format': precision.format.value}
    if not isinstance(precision, QuantIntent):
        raise TypeError(f'Unsupported precision type {type(precision)} when serializing backend IR.')
    return {
        'width': int(precision.width),
        'frac': int(precision.frac),
        'signed': bool(precision.signed),
        'rounding': precision.rounding.name,
        'saturation': precision.saturation.name,
    }


__all__ = ['dump_pipeline_ir']
