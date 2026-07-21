# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .aie_types import FLOAT_FORMATS, AIEDataType
from .ir import get_backend_context
from .quant_utils import apply_rounding, dtype_for_precision, handle_overflow

log = logging.getLogger(__name__)


def _is_float_format(fmt: str) -> bool:
    return (fmt or '') in FLOAT_FORMATS


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
      - ctx.ir.physical.plan['buffers']
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
            st = writer.get('staging')
            dtype = _dtype_from_plan(writer.get('dtype'))
            if st is None:
                raise RuntimeError(f'{tensor}: physical plan is missing graph-input staging data.')
            if dtype is None:
                raise RuntimeError(f'{tensor}: physical plan is missing graph-input dtype data.')

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
            st = reader.get('staging')
            dtype = _dtype_from_plan(reader.get('dtype'))
            if st is None:
                raise RuntimeError(f'{tensor}: physical plan is missing graph-output staging data.')
            if dtype is None:
                raise RuntimeError(f'{tensor}: physical plan is missing graph-output dtype data.')

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


def _dtype_from_plan(data: Dict | None) -> AIEDataType | None:
    if data is None:
        return None
    return AIEDataType.from_dict(data)


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

        if _is_float_format(p0.dtype.format):
            prepared[tensor] = np.asarray(arr, dtype=np.float32)
        elif quantize:
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

    is_float = np.issubdtype(np.asarray(values).dtype, np.floating)
    for idx, value in enumerate(values):
        if idx and idx % vals_per_line == 0:
            stream.write('\n')
        elif idx:
            stream.write(' ')
        stream.write(f'{float(value):e}' if is_float else str(int(value)))

    stream.write('\n')


def collect_outputs(output_dir: Path, sim_mode: str, layout: IOLayout) -> Dict[str, np.ndarray]:
    data_dir = Path(output_dir) / f'{sim_mode}simulator_output/data'
    outputs: Dict[str, np.ndarray] = {}

    for tensor, ports in layout.outputs.items():
        first = ports[0]
        first_tile = _read_output_file(data_dir / f'y_p{first.port}.txt', first)
        buf_dtype = np.float64 if _is_float_format(first.dtype.format) else np.int64
        out = np.zeros((first_tile.shape[0], *first.numpy_boundary_shape), dtype=buf_dtype)
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

    if _is_float_format(port.dtype.format):
        values = np.array([float(t) for t in clean], dtype=np.float64)
    else:
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
