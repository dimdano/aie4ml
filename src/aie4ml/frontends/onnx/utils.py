from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...aie_types import FloatFormat, FloatIntent, QuantIntent, RoundingMode, SaturationMode
from ...device_catalog import load_device_catalog
from ...ir import BackendPolicies
from ...ir.context import AIEBackendContext, DeviceSpec, ProjectConfig
from ...system_plan import normalize_pl_config
from ..common import register_default_traits


def require_onnx():
    try:
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as exc:
        raise ImportError('ONNX frontend requires the "onnx" Python package to be installed.') from exc
    return onnx, helper, numpy_helper


def node_name(node, index: int) -> str:
    name = (node.name or '').strip()
    return name if name else f'{node.op_type}_{index}'


def shape_from_value_info(value_info) -> Tuple[int, ...]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField('shape'):
        raise ValueError(f'{value_info.name}: tensor shape is missing.')
    dims = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField('dim_value'):
            raise ValueError(f'{value_info.name}: dynamic shapes are not supported.')
        dims.append(int(dim.dim_value))
    return tuple(dims)


def normalize_directives(name: str, raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f'{name}: layer directives must be a dict.')

    directives: Dict[str, Any] = {}

    if 'placement' in raw:
        placement_cfg = raw['placement']
        if not isinstance(placement_cfg, dict):
            raise TypeError(f'{name}: placement override must be a dict.')
        placement: Dict[str, int] = {}
        if 'col' in placement_cfg:
            placement['col'] = int(placement_cfg['col'])
        if 'row' in placement_cfg:
            placement['row'] = int(placement_cfg['row'])
        if placement:
            directives['placement'] = placement

    if 'microtiling' in raw:
        tiling_cfg = raw['microtiling']
        if not isinstance(tiling_cfg, dict):
            raise TypeError(f'{name}: tiling override must be a dict.')
        tiling: Dict[str, int] = {}
        for key in ('microtile_m', 'microtile_k', 'microtile_n'):
            if key in tiling_cfg:
                tiling[key] = int(tiling_cfg[key])
        if tiling:
            directives['microtiling'] = tiling

    if 'parallelism' in raw:
        parallel_cfg = raw['parallelism']
        if not isinstance(parallel_cfg, dict):
            raise TypeError(f'{name}: parallelism override must be a dict.')
        parallelism: Dict[str, int] = {}
        for key in ('cas_num', 'cas_length', 'parallel_factor'):
            if key in parallel_cfg:
                parallelism[key] = int(parallel_cfg[key])
        if parallelism:
            directives['parallelism'] = parallelism

    if 'io_route' in raw:
        io_route = raw['io_route']
        if not isinstance(io_route, dict):
            raise TypeError(f'{name}: io_route override must be a dict.')
        directives['io_route'] = dict(io_route)

    if 'approximation' in raw:
        directives['approximation'] = str(raw['approximation'])

    if 'hccs' in raw:
        hccs_cfg = raw['hccs']
        if not isinstance(hccs_cfg, dict):
            raise TypeError(f'{name}: hccs override must be a dict.')
        directives['hccs'] = dict(hccs_cfg)

    return directives


def resolve_project_name(model_path, model_proto, project_name: Optional[str]) -> str:
    if project_name:
        return project_name
    if model_path is not None:
        return Path(model_path).stem
    graph_name = str(getattr(model_proto.graph, 'name', '') or '').strip()
    if graph_name:
        return graph_name
    raise ValueError('project_name must be provided when lowering an in-memory ONNX model without graph.name.')


def create_context(config: Dict[str, Any], output_dir, project_name: str, stamp, custom_sources) -> AIEBackendContext:
    aie_cfg = dict(config.get('AIEConfig', {}) or {})
    part_name = aie_cfg.get('Part') or config.get('Part') or aie_cfg.get('Device')
    if not part_name:
        raise KeyError('ONNX frontend requires Part or AIEConfig.Part in the config dict.')

    catalog = load_device_catalog()
    device_entry = catalog.get(part_name, {}) or catalog.get(str(part_name).lower(), {})
    merged = dict(device_entry)
    merged.update(aie_cfg)
    if 'Generation' not in merged:
        merged['Generation'] = device_entry.get('Generation', '')

    device = DeviceSpec.from_config(str(part_name), merged)
    policies = BackendPolicies(
        fusion=dict(config.get('AIEFusionPolicy', {}) or {}),
        decomposition=dict(config.get('AIEDecompositionPolicy', {}) or {}),
        pack=dict(config.get('AIEPackPolicy', {}) or {}),
        cache=dict(config.get('AIECachePolicy', {}) or {}),
        tensors_have_batch=True,
    )
    project_config = ProjectConfig(
        output_dir=Path(output_dir),
        project_name=project_name,
        stamp=stamp,
        custom_sources=dict(custom_sources or {}),
    )
    resolved_aie_config = dict(merged)
    resolved_aie_config['Part'] = str(part_name)

    normalize_pl_config(resolved_aie_config)

    ctx = AIEBackendContext(
        device=device,
        policies=policies,
        project_config=project_config,
        aie_config=resolved_aie_config,
    )
    register_default_traits(ctx)
    return ctx


def initializer_map(graph, numpy_helper) -> Dict[str, np.ndarray]:
    return {init.name: np.asarray(numpy_helper.to_array(init)) for init in graph.initializer}


def input_maps(graph, initializer_names) -> Tuple[Dict[str, Tuple[int, ...]], Dict[str, int]]:
    shapes: Dict[str, Tuple[int, ...]] = {}
    elem_types: Dict[str, int] = {}
    for value_info in graph.input:
        if value_info.name in initializer_names:
            continue
        shapes[value_info.name] = shape_from_value_info(value_info)
        elem_types[value_info.name] = value_info.type.tensor_type.elem_type
    return shapes, elem_types


def scalar_tensor(initializers: Dict[str, np.ndarray], name: str, node_name: str) -> np.ndarray:
    if name not in initializers:
        raise ValueError(f'{node_name}: quantization parameter {name} must be an initializer.')
    arr = np.asarray(initializers[name])
    if arr.size != 1:
        raise ValueError(f'{node_name}: per-axis quantization is not supported for {name}.')
    return arr.reshape(())


def intent_from_qparams(
    initializers: Dict[str, np.ndarray],
    scale_name: str,
    zero_name: str,
    qdtype,
    node_name: str,
) -> QuantIntent:
    scale = float(scalar_tensor(initializers, scale_name, node_name))
    zero = scalar_tensor(initializers, zero_name, node_name)
    if int(zero) != 0:
        raise ValueError(f'{node_name}: zero_point must be 0 for symmetric quantization.')
    dtype = np.dtype(qdtype)
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f'{node_name}: only integer quantization is supported; got {dtype}.')
    if scale <= 0.0:
        raise ValueError(f'{node_name}: quantization scale must be positive.')
    log2_scale = np.log2(scale)
    rounded = round(float(log2_scale))
    if not np.isclose(log2_scale, rounded, atol=1e-7):
        raise ValueError(f'{node_name}: scale {scale} is not a power of two.')
    return QuantIntent(
        width=int(dtype.itemsize * 8),
        frac=int(-rounded),
        signed=bool(np.issubdtype(dtype, np.signedinteger)),
        rounding=RoundingMode.RND_CONV,
        saturation=SaturationMode.SAT,
    )


def dequantize_data(
    data: np.ndarray,
    initializers: Dict[str, np.ndarray],
    scale_name: str,
    zero_name: str,
    node_name: str,
) -> np.ndarray:
    scale = float(scalar_tensor(initializers, scale_name, node_name))
    zero = int(scalar_tensor(initializers, zero_name, node_name))
    return (np.asarray(data, dtype=np.float64) - float(zero)) * scale


def intent_from_initializer(data: np.ndarray, node_name: str):
    dtype = np.asarray(data).dtype
    if dtype == np.dtype(np.float32):
        return FloatIntent(width=32, format=FloatFormat.FP32)
    if str(dtype) == 'bfloat16':
        return FloatIntent(width=16, format=FloatFormat.BF16)
    if str(dtype) == 'float8_e4m3fn':
        return FloatIntent(width=8, format=FloatFormat.FP8_E4M3)
    raise ValueError(
        f'{node_name}: direct initializer inputs must be float32/bfloat16/fp8_e4m3, '
        f'or quantized via DequantizeLinear; got {dtype}.'
    )


def attr(node, name: str, default=None):
    _, helper, _ = require_onnx()
    for value in node.attribute:
        if value.name != name:
            continue
        return helper.get_attribute_value(value)
    return default
