# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""System-level (PL data mover + host) emission planning.

The AIE array project is always emitted. When the user selects ``target='hardware'`` on the
frontend, the project additionally emits the PL data mover, the v++ linker connectivity, and
the XRT host program so the design can run on a board (not just simulate the array).
``emits_system`` is the single gate consulted by the writer and ``AIEModel`` for that opt-in.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .ir import get_backend_context
from .passes.utils import sanitize_identifier

# A 512-bit DDR word carries 64 bytes; the PL data mover moves whole 512-bit words.
_DDR_WORD_BYTES = 64

# Compile-time cap on iterations the data mover preloads into PL URAM. Should ultimately be
# derived from available PL URAM (URAM budget / per-iteration footprint), not the AIE sim
# iteration count. Hardcoded until URAM-aware sizing lands.
_DEFAULT_MAX_N_ITER = 64


def emits_system(model_or_ctx) -> bool:
    """Return True when PL + host code should be emitted (``target='hardware'``).

    Defaults to AIE-only emission (``target='aie'``) when the key is absent.
    """
    ctx = get_backend_context(model_or_ctx)
    return str(ctx.aie_config.get('Target', 'aie')).lower() == 'hardware'


def _single_io_feat(ports_map: Dict[str, Any], direction: str, batch: int):
    """Return (per-sample feature count, element bytes) for a single graph IO tensor.

    Supports exactly one graph input and one graph output (multiple graph I/O tensors are
    not yet supported).
    """
    tensors = list(ports_map)
    if len(tensors) != 1:
        raise NotImplementedError(
            f'system I/O plan supports a single graph {direction} tensor; '
            f'got {len(tensors)} ({tensors}). Multiple graph {direction}s are not yet supported.'
        )
    port0 = ports_map[tensors[0]][0]
    total = int(math.prod(port0.numpy_boundary_shape))
    if total % int(batch) != 0:
        raise RuntimeError(f'graph {direction} {tensors[0]!r}: element count {total} is not divisible by batch {batch}.')
    return total // int(batch), int(port0.dtype.width) // 8


def _stream_words_512(batch: int, feat: int, elem_bytes: int, n_streams: int, direction: str) -> int:
    """512-bit words per stream per iteration; requires 512-bit + per-stream alignment."""
    total_bytes = int(batch) * int(feat) * int(elem_bytes)
    if total_bytes % (_DDR_WORD_BYTES * int(n_streams)) != 0:
        raise NotImplementedError(
            f'graph {direction}: {total_bytes} bytes is not a multiple of '
            f'{_DDR_WORD_BYTES} B/word * {n_streams} stream(s); 512-bit/per-stream padding is not yet supported.'
        )
    return total_bytes // (_DDR_WORD_BYTES * int(n_streams))


def build_system_io(model_or_ctx) -> Dict[str, Any]:
    """Project the resolved IR + physical plan into the flat variable bag the PL/host
    templates in ``templates/system/`` consume.

    Reuses :func:`aie4ml.simulation.build_io_layout` for per-PLIO-port boundary/dtype data and
    the dense/matmul execution entries for per-layer RTP (weight/bias) loading. Scope:
    single graph input + single graph output, 512-bit-aligned sizes.
    """
    from .simulation import build_io_layout

    ctx = get_backend_context(model_or_ctx)
    plan = ctx.ir.physical.plan
    if 'graph_input_count' not in plan or 'buffers' not in plan:
        raise RuntimeError('build_system_io requires a materialized physical plan; run the pipeline first.')

    layout = build_io_layout(ctx)
    n_ifm = int(plan['graph_input_count'])
    n_ofm = int(plan['graph_output_count'])
    batch = int(ctx.aie_config['BatchSize'])

    in_feat, in_bytes = _single_io_feat(layout.inputs, 'input', batch)
    out_feat, out_bytes = _single_io_feat(layout.outputs, 'output', batch)

    # Per-layer RTP (weight/bias) loading, in writer layer order (every node with an
    # execution entry increments the index, matching AIEProjectEmitter._collect_layers).
    layers = []
    layer_index = 0
    for node in ctx.ir.logical:
        inst = ctx.ir.execution.get(node.name)
        if inst is None:
            continue
        layer_index += 1
        artifacts = inst.variant.get_artifacts(inst)
        weights = next((a for a in artifacts if a.get('name') == 'weights'), None)
        if weights is None:
            continue  # param-less op (no kernel-resident RTP weights)
        bias = next((a for a in artifacts if a.get('name') == 'bias'), None)
        parallelism = getattr(inst.config, 'parallelism', None)
        if parallelism is None:
            raise RuntimeError(f'{node.name}: weight-bearing layer has no parallelism config.')
        inst_name = sanitize_identifier(node.name)
        layers.append(
            {
                'inst_name': inst_name,
                'weights_header': weights['filename'],
                'bias_header': bias['filename'] if bias else None,
                'weights_prefix': f"{weights['name']}_{inst_name}",
                'bias_prefix': f"{bias['name']}_{inst_name}" if bias else None,
                'wts_port': f"{weights['port']}{layer_index}",
                'bias_port': f"{bias['port']}{layer_index}" if bias else None,
                'cas_num': int(parallelism.cas_num),
                'cas_length': int(parallelism.cas_length),
            }
        )

    # Top-level cas_* describe the graph-output-producing (last weight) layer.
    cas_num = layers[-1]['cas_num'] if layers else 1
    cas_length = layers[-1]['cas_length'] if layers else 1
    if in_feat % n_ifm != 0:
        raise NotImplementedError(
            f'in_feat {in_feat} not divisible by n_ifm {n_ifm}; uneven input shard is not yet supported.'
        )
    if out_feat % cas_num != 0:
        raise NotImplementedError(
            f'out_feat {out_feat} not divisible by cas_num {cas_num}; uneven output shard is not yet supported.'
        )

    max_512_per_stream = max(
        _stream_words_512(batch, in_feat, in_bytes, n_ifm, 'input'),
        _stream_words_512(batch, out_feat, out_bytes, n_ofm, 'output'),
    )

    iterations = int(ctx.aie_config['Iterations'])
    return {
        'project_name': ctx.project_config.project_name,
        'platform': ctx.device.platform,
        'graph_name': 'dut',
        'pl_freq_hz': float(ctx.aie_config['PLClockFreqMHz']) * 1e6,
        'plio_width_bits': int(ctx.device.plio_width_bits),
        'n_ifm': n_ifm,
        'n_ofm': n_ofm,
        'batch': batch,
        'iterations': iterations,
        # MAX_N_ITER sizes the data mover's on-chip URAM preload buffers
        # (MAX_BIG_IN/OUT = MAX_512_PER_STREAM * MAX_N_ITER), so it must be driven by the PL
        # URAM budget, NOT the AIE sim Iterations. TODO: derive from device URAM capacity vs
        # max_512_per_stream; hardcoded for now.
        'max_n_iter': _DEFAULT_MAX_N_ITER,
        'in_feat': in_feat,
        'out_feat': out_feat,
        'in_feat_slice': in_feat // n_ifm,
        'out_feat_slice': out_feat // cas_num,
        'cas_num': cas_num,
        'cas_length': cas_length,
        'max_512_per_stream': max_512_per_stream,
        'layers': layers,
    }


# ---------------------------------------------------------------------------
# Host data.h generation (DDR-packed input) — target='hardware'
#
# The PL data mover (templates/firmware/pl/ddr_pl_aie_datamover.cpp.jinja) moves whole 512-bit
# DDR words and round-robin stripes them across the N per-direction PLIO streams
# (word i -> stream i % N). The host (host.cpp) replays this packed input each iteration.
# This produces `host/data.h` with the packed input and the IO word sizes.
# ---------------------------------------------------------------------------


def _pack_ports_to_ddr(port_tiles: List[np.ndarray], n_streams: int) -> np.ndarray:
    """Round-robin pack per-stream int8 tiles into the data mover's 512-bit DDR layout.

    Returns a little-endian uint32 array (the host buffer element type). Mirrors the data
    mover: contiguous DDR word i is owned by stream i % n_streams.
    """
    word_blocks = []
    for tile in port_tiles:
        flat = np.ascontiguousarray(tile).astype(np.int8).reshape(-1)
        if flat.size % _DDR_WORD_BYTES != 0:
            raise NotImplementedError(
                f'stream tile of {flat.size} B is not a multiple of {_DDR_WORD_BYTES} B; 512-bit padding is not yet supported.'
            )
        word_blocks.append(flat.reshape(-1, _DDR_WORD_BYTES))
    if len(word_blocks) != int(n_streams):
        raise RuntimeError(f'expected {n_streams} stream tiles, got {len(word_blocks)}.')
    words = word_blocks[0].shape[0]
    if any(block.shape[0] != words for block in word_blocks):
        raise NotImplementedError('uneven per-stream word counts; relay/padding is not yet supported.')

    mem = np.empty((words * int(n_streams), _DDR_WORD_BYTES), dtype=np.int8)
    for stream, block in enumerate(word_blocks):
        mem[stream :: int(n_streams)] = block
    return np.frombuffer(mem.reshape(-1).tobytes(), dtype='<u4')


def pack_host_data(model_or_ctx, X=None):
    """Return (ifm_packed, ofm_size_words) for one iteration, in the data mover DDR layout.

    Packs the quantized graph input into the DDR layout the host replays, and reports the
    output size (in 32-bit words) the host needs to size its output buffer. Both are
    graph-agnostic (no forward pass). X defaults to a deterministic pseudo-random input.
    """
    from .simulation import _extract_port_tile, build_io_layout, prepare_inputs

    ctx = get_backend_context(model_or_ctx)
    layout = build_io_layout(ctx)
    if len(layout.inputs) != 1 or len(layout.outputs) != 1:
        raise NotImplementedError('host data.h supports a single graph input and output.')

    in_tensor = next(iter(layout.inputs))
    in_ports = layout.inputs[in_tensor]
    out_ports = layout.outputs[next(iter(layout.outputs))]
    batch, in_feat = in_ports[0].numpy_boundary_shape

    if X is None:
        X = np.random.default_rng(0).random((int(batch), int(in_feat)), dtype=np.float64) * 2.0 - 1.0

    prepared = prepare_inputs(layout, X, iterations=1, quantize=True)[in_tensor]  # (1, *boundary) int
    in_tiles = [_extract_port_tile(prepared, p)[0].reshape(-1) for p in in_ports]
    ifm_packed = _pack_ports_to_ddr(in_tiles, len(in_ports))

    out_total_bytes = int(np.prod(out_ports[0].numpy_boundary_shape)) * (int(out_ports[0].dtype.width) // 8)
    ofm_size_words = out_total_bytes // 4
    return ifm_packed, ofm_size_words


def _emit_uint32_array(name: str, size_macro: str, values: np.ndarray) -> str:
    body = ', '.join(str(int(v) & 0xFFFFFFFF) for v in values)
    return f'static const uint32_t {name}[{size_macro}] = {{ {body} }};\n'

# Direct method to write host/data.h TODO: Should be moved to jinja template
def write_host_data_header(model_or_ctx, output_dir, X=None) -> Path:
    """Write ``host/data.h`` (ifm_packed + IO word sizes) consumed by host.cpp."""
    ifm_packed, ofm_size_words = pack_host_data(model_or_ctx, X)
    host_dir = Path(output_dir) / 'host'
    host_dir.mkdir(parents=True, exist_ok=True)
    dest = host_dir / 'data.h'
    text = (
        '#pragma once\n'
        '// GENERATED by aie4ml (DDR-packed input). Do not edit by hand.\n'
        '#include <cstdint>\n\n'
        f'static const int IFM_SIZE_WORDS = {len(ifm_packed)};\n'
        f'static const int OFM_SIZE_WORDS = {ofm_size_words};\n\n'
        + _emit_uint32_array('ifm_packed', 'IFM_SIZE_WORDS', ifm_packed)
    )
    dest.write_text(text)
    return dest
