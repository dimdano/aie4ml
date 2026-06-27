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
from typing import Any, Dict, List

import numpy as np

from .ir import get_backend_context
from .passes.utils import sanitize_identifier

# Bytes in one 512-bit DDR/AXI word -- the unit the PL data mover transfers.
# This is a transport constant (matches the kernel's ap_uint<512> m_axi word); it is
# independent of the element dtype (int8/int16/int32 just change how many elements fit
# per word, not the word size).
_DDR_WORD_BYTES = 64

# Compile-time cap on iterations the data mover preloads into PL URAM. Should ultimately be
# derived from available PL URAM (URAM budget / per-iteration footprint), not the AIE sim
# iteration count. Hardcoded until URAM-aware sizing lands.
_DEFAULT_MAX_N_ITER = 64

# PL data-mover template locations. The benchmark mover lives under pl/benchmark/; the
# deployment movers (memory_stream / external_stream) live under pl/deployment/.
_BENCHMARK_KERNEL = 'ddr_pl_aie_datamover'
_BENCHMARK_TEMPLATE_DIR = 'pl/benchmark'
_DEPLOYMENT_TEMPLATE_DIR = 'pl/deployment'


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


def _kernel_entry(name: str, template_dir: str) -> Dict[str, str]:
    """One PL kernel for the writer (which templates to render) and the Makefile (.xo)."""
    return {
        'name': name,
        'cpp_template': f'{template_dir}/{name}.cpp.jinja',
        'cfg_template': f'{template_dir}/{name}.cfg.jinja',
    }


def build_pl_plan(mode: str, n_ifm: int, n_ofm: int, enable_pl_timing: bool) -> Dict[str, Any]:
    """Describe the PL data path for a given ``PLDataMoverMode``: which kernels to emit,
    the v++ linker connectivity (``nk=`` / ``sc=``), and the host's timing wiring.

    This is the single source of truth consumed by:
      * the writer            -> ``plan['kernels']`` (which .cpp/.cfg templates to render),
      * ``Makefile.jinja``    -> kernel ``.xo`` list + per-kernel build,
      * ``system.cfg.jinja``  -> ``plan['nk']`` / ``plan['sc']`` connectivity,
      * ``host.cpp.jinja``    -> ``plan['host']`` (timing CU + ``cycles_*`` register names).

    Modes:
      ``benchmark``     single combined ``ddr_pl_aie_datamover`` CU (today's design);
      ``memory_stream`` split ``mm2s`` + ``s2mm`` double-buffered CUs, plus a shared
                        ``tick_gen`` timer CU when PL timing is on (separate CUs have
                        independent ``ap_start``, so one timer gives a common time base);
      ``external_stream`` not yet emitted (direct PLIO<->external PL wiring + traffic-gen).
    """
    mode = str(mode).lower()

    if mode == 'benchmark':
        # Benchmark is a cycle-accurate measurement harness -- the PL timers are its
        # whole point, so force them ON regardless of EnablePLTiming.
        if not enable_pl_timing:
            print("[aie4ml] PLDataMoverMode='benchmark' forces PL timers ON "
                  "(overriding EnablePLTiming=False).")
            enable_pl_timing = True
        name = _BENCHMARK_KERNEL
        sc = [f'{name}.s_out_{s}:ai_engine_0.PLIO_ifm_{s}' for s in range(n_ifm)]
        sc += [f'ai_engine_0.PLIO_ofm_{s}:{name}.s_in_{s}' for s in range(n_ofm)]
        # The combined CU owns tick_gen internally; the host reads its 7 cycles_* regs.
        return {
            'mode': mode,
            'enable_pl_timing': enable_pl_timing,
            'kernels': [_kernel_entry(name, _BENCHMARK_TEMPLATE_DIR)],
            'nk': [f'{name}:1:{name}'],
            'sc': sc,
            'host': {
                'timing_kernel': name,
                'cycles': ['preload_done', 'first_send', 'last_send', 'first_recv',
                           'last_recv', 'compute_done', 'total'],
            },
        }

    if mode == 'memory_stream':
        ifm_k, ofm_k = 'mm2s', 's2mm'
        kernels = [_kernel_entry(ifm_k, _DEPLOYMENT_TEMPLATE_DIR),
                   _kernel_entry(ofm_k, _DEPLOYMENT_TEMPLATE_DIR)]
        nk = [f'{ifm_k}:1:{ifm_k}', f'{ofm_k}:1:{ofm_k}']
        sc = [f'{ifm_k}.s_out_{s}:ai_engine_0.PLIO_ifm_{s}' for s in range(n_ifm)]
        sc += [f'ai_engine_0.PLIO_ofm_{s}:{ofm_k}.s_in_{s}' for s in range(n_ofm)]
        host = {'timing_kernel': None, 'cycles': []}
        if enable_pl_timing:
            timer = 'tick_gen'
            kernels.append(_kernel_entry(timer, _DEPLOYMENT_TEMPLATE_DIR))
            nk.append(f'{timer}:1:{timer}')
            # PL-to-PL event pulses give tick_gen a single time base across the two CUs.
            sc += [
                f'{ifm_k}.ev_first_send:{timer}.ev_first_send',
                f'{ifm_k}.ev_last_send:{timer}.ev_last_send',
                f'{ofm_k}.ev_first_recv:{timer}.ev_first_recv',
                f'{ofm_k}.ev_last_recv:{timer}.ev_last_recv',
                f'{ofm_k}.ev_done:{timer}.ev_done',
            ]
            host = {'timing_kernel': timer,
                    'cycles': ['first_send', 'last_send', 'first_recv', 'last_recv', 'total']}
        return {'mode': mode, 'enable_pl_timing': enable_pl_timing,
                'kernels': kernels, 'nk': nk, 'sc': sc, 'host': host}

    if mode == 'external_stream':
        raise NotImplementedError(
            "PLDataMoverMode 'external_stream' is not yet emitted by the compiler "
            '(direct PLIO<->external PL AXI-stream wiring + traffic-gen test pending).'
        )
    raise ValueError(f'unknown PLDataMoverMode {mode!r}.')


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
    if len(layout.inputs) != 1 or len(layout.outputs) != 1:
        raise RuntimeError(
            f'system I/O plan supports a single graph tensor; '
            f'got {len(layout.inputs)} and ({layout.outputs}). Multiple graph are not yet supported.'
        )

    in_feat, in_bytes = _single_io_feat(layout.inputs, 'input', batch)
    out_feat, out_bytes = _single_io_feat(layout.outputs, 'output', batch)
    # Per-PLIO-tile feature slices come from the GRAPH BOUNDARY ports -- the single graph
    # input and single graph output (already validated to be 1 each above). 
    gin_port = next(iter(layout.inputs.values()))[0]  
    gout_port = next(iter(layout.outputs.values()))[0]
    in_feat_slice = gin_port.tiling_dimension[gin_port.slice_dimension]
    out_feat_slice = gout_port.tiling_dimension[gout_port.slice_dimension]

    # Per-layer RTP artifacts (weights/bias/...). layer_index counts every executed node
    # (incl. param-less activations) so the RTP port suffix matches the app.cpp graph index.
    layers = []
    layer_index = 0
    for node in ctx.ir.logical:
        inst = ctx.ir.execution.get(node.name)
        if inst is None:
            continue
        layer_index += 1
        artifacts = inst.variant.get_artifacts(inst)
        if not artifacts:
            continue  # param-less op (e.g. activation) -- no kernel-resident RTP
        parallelism = getattr(inst.config, 'parallelism', None)
        if parallelism is None:
            raise RuntimeError(f'{node.name}: RTP-bearing layer has no parallelism config.')
        inst_name = sanitize_identifier(node.name)
        layers.append({
            'inst_name': inst_name,
            'cas_num': int(parallelism.cas_num),
            'cas_length': int(parallelism.cas_length),
            'artifacts': [
                {
                    'name': a['name'],
                    'kind': a['kind'],                     # '1d' | '2d' -> RTP loop shape
                    'header': a['filename'],               # generated header to #include
                    'prefix': f"{a['name']}_{inst_name}",  # C symbol base (matches header)
                    'port': f"{a['port']}{layer_index}",   # ADF RTP port (name + layer idx, == app.cpp Lidx)
                }
                for a in artifacts
            ],
        })
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
    # HLS storage impl for the data mover preload buffers: URAM (default) or BRAM.
    pl_mem_impl = 'BRAM' if str(ctx.aie_config.get('PLMemory', 'uram')).lower() == 'bram' else 'URAM'
    # PL data-path style (benchmark single CU vs split deployment movers). The plan tells
    # the writer which kernels to render and the templates how to wire them, and resolves
    # the effective PL timing (benchmark forces the timers on). Built once here and handed
    # back via system_io['pl_plan'] so the Makefile render can reuse it.
    pl_data_mover_mode = str(ctx.aie_config.get('PLDataMoverMode', 'benchmark')).lower()
    pl_plan = build_pl_plan(pl_data_mover_mode, n_ifm, n_ofm,
                            bool(ctx.aie_config.get('EnablePLTiming', True)))
    enable_pl_timing = pl_plan['enable_pl_timing']
    return {
        'project_name': ctx.project_config.project_name,
        'platform': ctx.device.platform,
        'graph_name': 'dut',
        'pl_freq_hz': float(ctx.aie_config['PLClockFreqMHz']) * 1e6,
        'plio_width_bits': int(ctx.device.plio_width_bits),
        'pl_mem_impl': pl_mem_impl,
        'enable_pl_timing': enable_pl_timing,
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
        'in_feat_slice': in_feat_slice,
        'out_feat_slice': out_feat_slice,
        'cas_num': cas_num,
        'cas_length': cas_length,
        'max_512_per_stream': max_512_per_stream,
        'layers': layers,
        'pl_data_mover_mode': pl_data_mover_mode,
        'pl_plan': pl_plan,
    }

# ---------------------------------------------------------------------------
# Host data.h generation (DDR-packed input) — target='hardware'
#
# The PL data movers (templates/firmware/pl/benchmark/ or pl/deployment/) move whole 512-bit
# DDR words and round-robin stripe them across the N per-direction PLIO streams
# (word i -> stream i % N). The host (host.cpp) replays this packed input each iteration.
# This produces `host/data.h` with the packed input and the IO word sizes.
# ---------------------------------------------------------------------------


def _pack_ports_to_ddr(port_tiles: List[np.ndarray], n_streams: int) -> np.ndarray:
    """Round-robin pack per-stream tiles into the data mover's 512-bit DDR layout.

    Packs each tile's RAW STORAGE BYTES (dtype-agnostic -- the PL data mover moves whole
    512-bit words and is type-agnostic; int8/int16/int32 just change how many elements fit
    per word). Returns a little-endian uint32 array (the host buffer element type), with
    contiguous DDR word i owned by stream i % n_streams. Assumes little-endian byte order
    (native on x86/Versal), consistent with the kernel's word reads.
    """
    word_blocks = []
    for tile in port_tiles:
        raw = np.frombuffer(np.ascontiguousarray(tile).tobytes(), dtype=np.uint8)
        if raw.size % _DDR_WORD_BYTES != 0:
            raise NotImplementedError(
                f'stream tile of {raw.size} B is not a multiple of {_DDR_WORD_BYTES} B; '
                f'512-bit padding is not yet supported.'
            )
        word_blocks.append(raw.reshape(-1, _DDR_WORD_BYTES))
    if len(word_blocks) != int(n_streams):
        raise RuntimeError(f'expected {n_streams} stream tiles, got {len(word_blocks)}.')
    words = word_blocks[0].shape[0]
    if any(block.shape[0] != words for block in word_blocks):
        raise NotImplementedError('uneven per-stream word counts; relay/padding is not yet supported.')

    mem = np.empty((words * int(n_streams), _DDR_WORD_BYTES), dtype=np.uint8)
    for stream, block in enumerate(word_blocks):
        mem[stream :: int(n_streams)] = block
    return np.frombuffer(mem.tobytes(), dtype='<u4')


def _check_storage_width(port, tile) -> None:
    """Fail hard if the prepared element width doesn't match the port's byte-aligned dtype.

    The packing is dtype-aware via raw bytes, but only for byte-aligned formats whose
    prepared NumPy dtype matches the boundary width (int8/int16/int32). Sub-byte widths or
    formats whose storage differs (e.g. bfloat16 prepared as float32) are rejected rather
    than silently corrupted.
    """
    if int(port.dtype.width) % 8 != 0:
        raise NotImplementedError(
            f'graph IO dtype {port.dtype.format!r} ({port.dtype.width}-bit) is not byte-aligned; '
            f'sub-byte PLIO packing is not supported.'
        )
    want = int(port.dtype.width) // 8
    got = int(tile.dtype.itemsize)
    if got != want:
        raise NotImplementedError(
            f'graph input {port.tensor!r} port {port.port}: prepared element is {got} B but boundary '
            f'dtype {port.dtype.format!r} is {want} B; dtype-exact packing for this format is unsupported.'
        )


def pack_host_data(model_or_ctx, X=None):
    """Return (ifm_packed, ofm_size_words) for one iteration, in the data mover DDR layout.

    Packs the quantized graph input into the DDR layout the host replays, and reports the
    output size (in 32-bit words) the host needs to size its output buffer. Both are
    graph-agnostic (no forward pass). X defaults to a deterministic pseudo-random input.

    dtype-aware: ``prepare_inputs`` quantizes X to the port's storage NumPy dtype
    (``dtype_for_precision(width, signed)`` for ints, float32 for floats) and the raw bytes
    of that are packed -- the same storage-dtype contract the weights path uses. n-D
    boundaries are handled via the full ``numpy_boundary_shape``.
    """
    from .simulation import _extract_port_tile, build_io_layout, prepare_inputs

    ctx = get_backend_context(model_or_ctx)
    layout = build_io_layout(ctx)
    in_tensor = next(iter(layout.inputs))
    in_ports = layout.inputs[in_tensor]
    out_port0 = layout.outputs[next(iter(layout.outputs))][0]

    boundary = in_ports[0].numpy_boundary_shape  # full n-D shape; no (batch, feat) assumption

    if X is None:
        X = np.random.default_rng(0).random(boundary, dtype=np.float64) * 2.0 - 1.0

    prepared = prepare_inputs(layout, X, iterations=1, quantize=True)[in_tensor]  # (1, *boundary)
    in_tiles = []
    for p in in_ports:
        tile = _extract_port_tile(prepared, p)[0]  # this port's slice, storage dtype, n-D
        _check_storage_width(p, tile)
        in_tiles.append(tile)
    ifm_packed = _pack_ports_to_ddr(in_tiles, len(in_ports))

    out_total_bytes = int(np.prod(out_port0.numpy_boundary_shape)) * (int(out_port0.dtype.width) // 8)
    if out_total_bytes % 4 != 0:
        raise NotImplementedError(
            f'graph output is {out_total_bytes} B, not a multiple of 4 B (uint32 host buffer).'
        )
    ofm_size_words = out_total_bytes // 4
    return ifm_packed, ofm_size_words


def host_data_context(model_or_ctx, X=None) -> Dict[str, Any]:
    """Prepare the ``host/data.h`` template context: the DDR-packed graph input and
    the IO word sizes the host needs.

    Python only prepares the values (masked to uint32); the writer renders
    ``templates/firmware/host/data.h.jinja``.
    """
    ifm_packed, ofm_size_words = pack_host_data(model_or_ctx, X)
    return {
        'ifm_packed': [int(v) & 0xFFFFFFFF for v in ifm_packed],
        'ifm_size_words': int(len(ifm_packed)),
        'ofm_size_words': int(ofm_size_words),
    }
