# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from .aie_types import AIEDataType
from .ir import get_backend_context
from .passes.quant import apply_rounding, dtype_for_precision, handle_overflow

log = logging.getLogger(__name__)


def read_aie_report(model_or_path: Union[object, str, Path]) -> Dict:
    model = None
    if hasattr(model_or_path, 'config'):
        model = model_or_path
        output_dir = Path(model.config.get_output_dir())
    else:
        output_dir = Path(model_or_path)
    output_dir = output_dir.resolve()

    ii_info = _analyze_aie_out_interval(output_dir)
    graph_info = _read_aie_graph_stats(output_dir)
    report = {}

    global_ii = ii_info.get('global', {})
    if global_ii and model is not None:
        ops_per_inf = compute_ops(model)
        batch_size = _infer_batch_size(model)
        total_ops = ops_per_inf * batch_size
        report['throughput'] = {
            'Avg_GOPs': round((total_ops / global_ii['avg_ns']), 3),
            'Min_GOPs': round((total_ops / global_ii['min_ns']), 3),
            'Max_GOPs': round((total_ops / global_ii['max_ns']), 3),
        }

    report['output_interval'] = ii_info
    report['AIE_info'] = graph_info

    return report


def _read_aie_graph_stats(output_dir: Path) -> Dict:
    report_path = output_dir / 'Work' / 'reports' / 'app_mapping_analysis_report.txt'

    if report_path.exists():
        try:
            with open(report_path) as f:
                text = f.read().strip('\n')
            return text.splitlines()
        except Exception as e:
            return f'error Failed to read AIE graph report: {e} in {str(report_path)}'

    return 'No AIE graph report found. Run AIE hardware compilation to generate it.'


def _analyze_aie_out_interval(output_dir: Path) -> Dict:
    data_dir = output_dir / 'aiesimulator_output' / 'data'

    if not data_dir.exists():
        return {}

    per_file = {}
    all_lat = []

    for fp in sorted(data_dir.glob('y_p*.txt')):
        lst = _parse_timing(fp)
        if lst:
            per_file[fp.name] = {
                'min_ns': round(min(lst), 3),
                'max_ns': round(max(lst), 3),
                'avg_ns': round(sum(lst) / len(lst), 3),
                'samples': len(lst),
            }
            all_lat.extend(lst)

    if not all_lat:
        return {}

    return {
        'global': {
            'min_ns': round(min(all_lat), 3),
            'max_ns': round(max(all_lat), 3),
            'avg_ns': round(sum(all_lat) / len(all_lat), 3),
            'samples': len(all_lat),
        },
        'per_port': per_file,
    }


def _parse_timing(path: Path) -> List[float]:
    """Return TLAST-to-TLAST intervals (in nanoseconds)."""
    regex = re.compile(r'^T\s+(\d+)\s*(ps|ns|us|ms|s)', re.IGNORECASE)

    lat = []
    last_tlast_time = None
    current_time = None

    with open(path) as f:
        for line in f:
            line = line.strip()

            m = regex.match(line)
            if m:
                val, unit = m.groups()
                current_time = _convert_to_ns(int(val), unit)
                continue

            if 'TLAST' in line.upper():
                if last_tlast_time is not None and current_time is not None:
                    dt = current_time - last_tlast_time
                    if dt >= 0:
                        lat.append(dt)
                last_tlast_time = current_time

    return lat


def _convert_to_ns(value: int, unit: str) -> float:
    if unit == 'ps':
        return value / 1000
    if unit == 'ns':
        return value
    if unit == 'us':
        return value * 1000
    if unit == 'ms':
        return value * 1_000_000
    if unit == 's':
        return value * 1_000_000_000
    raise ValueError(f'Unknown time unit: {unit}')


def compute_ops(model):
    ctx = get_backend_context(model)
    ops = 0
    for node in ctx.ir.logical:
        if node.op_type == 'dense':
            n_in = int(node.metadata['n_in'])
            n_out = int(node.metadata['n_out'])
            ops += 2 * n_in * n_out
    return ops


def _infer_batch_size(model) -> int:
    ctx = get_backend_context(model)
    inputs = ctx.ir.logical.graph_inputs()
    if not inputs:
        raise RuntimeError('Cannot infer batch size: logical graph has no inputs.')
    shape = inputs[0].shape
    if not shape:
        raise RuntimeError('Cannot infer batch size: input tensor shape missing.')
    return int(shape[0])


@dataclass
class IOPortLayout:
    direction: str
    port: int
    tensor: str
    descriptor: Dict
    staging: Dict
    dtype: AIEDataType

    @property
    def rank(self) -> int:
        return len(self.staging['io_boundary_dimension'])

    @property
    def slice_dimension(self) -> int:
        return int(self.staging['slice_dimension'])

    @property
    def io_boundary_dimension(self) -> List[int]:
        return [int(x) for x in self.staging['io_boundary_dimension']]

    @property
    def io_tiling_dimension(self) -> List[int]:
        return [int(x) for x in self.staging['io_tiling_dimension']]

    @property
    def offset(self) -> List[int]:
        # Host-visible tensor slicing must use the kernel staging descriptor.
        # Physical plan offsets may be shard-rebased for memtile units>1.
        return [int(x) for x in self.staging['offset']]

    @property
    def tiling_dimension(self) -> List[int]:
        # Files stream IO tiles, not kernel tiles.
        return [int(x) for x in self.staging['io_tiling_dimension']]

    @property
    def numpy_boundary_shape(self) -> Tuple[int, ...]:
        return tuple(self.io_boundary_dimension[::-1])

    @property
    def numpy_tile_shape(self) -> Tuple[int, ...]:
        return tuple(self.tiling_dimension[::-1])


@dataclass
class IOLayout:
    inputs: Dict[str, List[IOPortLayout]]
    outputs: Dict[str, List[IOPortLayout]]

    def input_tensors(self) -> List[str]:
        return list(self.inputs.keys())

    def output_tensors(self) -> List[str]:
        return list(self.outputs.keys())


def build_io_layout(model) -> IOLayout:
    """
    Build a canonical per-port IO layout strictly from:
      - ctx.ir.physical.plan['buffers'] (external endpoint discovery/mapping: ifm[x]/ofm[x])
      - ctx.ir.kernels (kernel instances + staging descriptors + numeric specs)
    """
    ctx = get_backend_context(model)
    plan = ctx.ir.physical.plan
    buffers = plan['buffers']

    inputs: Dict[str, List[IOPortLayout]] = {}
    outputs: Dict[str, List[IOPortLayout]] = {}

    for buf in buffers:
        tensor = buf['tensor']

        for writer in buf['writers']:
            if writer['source_type'] != 'plio':
                continue
            if writer['source_endpoint']['name'] != 'ifm':
                continue
            port = int(writer['source_endpoint']['port'])

            reader = next(r for r in buf['readers'] if r['target_type'] == 'kernel')
            inst = ctx.ir.kernels.get(reader['target_endpoint']['kernel'])

            st = inst.variant.describe_input_staging(
                inst.node,
                inst.attributes,
                tensor,
                port,
                None,
                None,
                None,
            )
            dtype = inst.attributes.numeric['input']

            inputs.setdefault(tensor, []).append(
                IOPortLayout(
                    direction='input',
                    port=port,
                    tensor=tensor,
                    descriptor=writer['descriptor'],
                    staging=st,
                    dtype=dtype,
                )
            )

        for reader in buf['readers']:
            if reader.get('target_type') != 'plio':
                continue
            if reader['target_endpoint']['name'] != 'ofm':
                continue
            port = int(reader['target_endpoint']['port'])

            writer = next(w for w in buf['writers'] if w['source_type'] == 'kernel')
            inst = ctx.ir.kernels.get(writer['source_endpoint']['kernel'])

            st = inst.variant.describe_output_staging(
                inst.node,
                inst.attributes,
                tensor,
                port,
                None,
                None,
            )
            dtype = inst.attributes.numeric['output']

            outputs.setdefault(tensor, []).append(
                IOPortLayout(
                    direction='output',
                    port=port,
                    tensor=tensor,
                    descriptor=reader['descriptor'],
                    staging=st,
                    dtype=dtype,
                )
            )

    for t in inputs:
        inputs[t] = sorted(inputs[t], key=lambda p: p.port)
    for t in outputs:
        outputs[t] = sorted(outputs[t], key=lambda p: p.port)

    return IOLayout(inputs=inputs, outputs=outputs)


def prepare_inputs(layout: IOLayout, X, iterations: int, quantize: bool = True) -> Dict[str, np.ndarray]:
    if len(layout.inputs) == 1 and not isinstance(X, dict):
        tensor = next(iter(layout.inputs.keys()))
        X = {tensor: X}

    prepared: Dict[str, np.ndarray] = {}

    for tensor, ports in layout.inputs.items():
        p0 = ports[0]
        expected = p0.numpy_boundary_shape
        arr = np.asarray(X[tensor])

        if tuple(arr.shape) == expected:
            arr = np.repeat(arr[np.newaxis, ...], iterations, axis=0)
        elif arr.ndim == len(expected) + 1 and tuple(arr.shape[1:]) == expected and arr.shape[0] == iterations:
            pass
        elif arr.ndim == len(expected) + 1 and tuple(arr.shape[1:]) == expected and arr.shape[0] == 1:
            arr = np.repeat(arr, iterations, axis=0)
        else:
            raise ValueError(
                f'{tensor}: expected shape {expected}, (1, *{expected}) or ({iterations}, *{expected}); '
                f'got {tuple(arr.shape)}'
            )

        if tuple(arr.shape[1:]) != expected:
            raise ValueError(f'{tensor}: expected input shape {expected}, got {tuple(arr.shape[1:])}')

        if quantize:
            prepared[tensor] = _quantize_to_int(
                arr,
                dtype=p0.dtype,
            )
        else:
            if not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(f'{tensor}: quantize=False requires integer inputs')
            prepared[tensor] = arr.astype(dtype_for_precision(p0.dtype.width, p0.dtype.signed), copy=False)

    return prepared


def write_input_files(output_dir: Path, layout: IOLayout, prepared_inputs: Dict[str, np.ndarray], plio_width_bits: int):
    data_dir = Path(output_dir) / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    for tensor, ports in layout.inputs.items():
        data = prepared_inputs[tensor]
        for p in ports:
            vals_per_line = max(1, int(plio_width_bits) // int(p.dtype.width))
            tile = _extract_port_tile(data, p)
            file_path = data_dir / f'ifm_c{p.port}.txt'
            with open(file_path, 'w') as handle:
                _write_values(handle, tile.flatten(order='C'), vals_per_line)


def _write_values(stream, values, vals_per_line):
    if vals_per_line <= 0:
        vals_per_line = len(values)

    for idx, value in enumerate(values):
        if idx and idx % vals_per_line == 0:
            stream.write('\n')
        elif idx:
            stream.write(' ')
        stream.write(str(int(value)))

    stream.write('\n')


def collect_outputs(output_dir: Path, sim_mode: str, layout: IOLayout) -> Dict[str, np.ndarray]:
    data_dir = Path(output_dir) / f'{sim_mode}simulator_output/data'
    outputs: Dict[str, np.ndarray] = {}

    for tensor, ports in layout.outputs.items():
        first = ports[0]
        first_tile = _read_output_file(data_dir / f'y_p{first.port}.txt', first)
        out = np.zeros((first_tile.shape[0], *first.numpy_boundary_shape), dtype=np.int64)
        _insert_port_tile(out, first_tile, first)

        for p in ports[1:]:
            tile = _read_output_file(data_dir / f'y_p{p.port}.txt', p)
            _insert_port_tile(out, tile, p)

        outputs[tensor] = out

    return outputs


def dequantize_outputs(layout: IOLayout, outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for tensor, arr in outputs.items():
        p0 = layout.outputs[tensor][0]
        frac = max(0, int(p0.dtype.frac))
        if frac == 0:
            out[tensor] = arr.astype(np.float64, copy=False)
        else:
            out[tensor] = arr.astype(np.float64, copy=False) / float(1 << frac)
    return out


def _quantize_to_int(data: np.ndarray, dtype: AIEDataType) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(dtype_for_precision(dtype.width, dtype.signed), copy=False)

    scale = 1 << max(0, int(dtype.frac))
    scaled = data * float(scale)
    rounded = apply_rounding(scaled, dtype.rounding)
    integers = rounded.astype(np.int64, copy=False)
    clipped = handle_overflow(integers, int(dtype.width), bool(dtype.signed), dtype.saturation)
    return clipped.astype(dtype_for_precision(dtype.width, dtype.signed), copy=False)


def _extract_port_tile(data: np.ndarray, port: IOPortLayout) -> np.ndarray:
    rank = port.rank
    tile = np.zeros((data.shape[0], *port.numpy_tile_shape), dtype=data.dtype)

    src_slices = [slice(None)] * (rank + 1)
    dst_slices = [slice(None)] * (rank + 1)
    for d in range(rank):
        axis = rank - d
        start = int(port.offset[d])
        size = int(port.tiling_dimension[d])
        bound = int(port.io_boundary_dimension[d])
        take = min(size, max(0, bound - start))

        src_slices[axis] = slice(start, start + take)
        dst_slices[axis] = slice(0, take)

    tile[tuple(dst_slices)] = data[tuple(src_slices)]
    return tile


def _insert_port_tile(out: np.ndarray, tile: np.ndarray, port: IOPortLayout) -> None:
    rank = port.rank
    dst_slices = [slice(None)] * (rank + 1)
    src_slices = [slice(None)] * (rank + 1)
    for d in range(rank):
        axis = rank - d
        start = int(port.offset[d])
        size = int(port.tiling_dimension[d])
        bound = int(port.io_boundary_dimension[d])
        take = min(size, max(0, bound - start))

        dst_slices[axis] = slice(start, start + take)
        src_slices[axis] = slice(0, take)

    out[tuple(dst_slices)] = tile[tuple(src_slices)]


def _read_output_file(path: Path, port: IOPortLayout) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f'Expected simulator output {path} not found.')

    tokens = path.read_text().split()
    clean: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.upper() == 'TLAST':
            i += 1
            continue
        if tok.upper() == 'T' and i + 2 < len(tokens):
            i += 3
            continue
        clean.append(tok)
        i += 1

    values = np.array([int(t) for t in clean], dtype=np.int64)
    per_iter = int(np.prod(port.numpy_tile_shape, dtype=np.int64))
    iters = values.size // per_iter
    values = values[: iters * per_iter]
    return values.reshape(iters, *port.numpy_tile_shape)


def run_simulation_target(output_dir, make_target):
    cmd = ['make', make_target]
    result = subprocess.run(cmd, cwd=Path(output_dir), text=True)
    if result.returncode != 0:
        raise RuntimeError(f'Make target "{make_target}" failed in {output_dir}.')
